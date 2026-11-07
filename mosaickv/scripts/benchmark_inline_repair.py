#!/usr/bin/env python3
"""Finite paired runtime verification; no fresh data or paper promotion."""

import argparse
import gc
import json
import os
import random
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

SOURCE = Path(__file__).resolve().parents[2]
GATE = Path("/scratch/djy8hg/runs/evicastkv/inline-first-step-repair-20261107-v1")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v3")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("package import does not belong to the immutable source worktree")
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable paired plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for path, value in plan["bound_files"].items():
        if sha(Path(path)) != value:
            raise ValueError("immutable input or validation evidence changed")
    return plan


def prepare(root, indexed_graph=False, graph_gate_root=None):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable paired root required")
    decision = read(GATE / "analysis/decision.json")
    if not decision["execution_passed"]:
        raise ValueError("bitwise inline/forced parity gate must pass before timing")
    if indexed_graph:
        graph_gate_root = graph_gate_root or Path(
            "/scratch/djy8hg/runs/evicastkv/indexed-positional-graph-20261107-v1"
        )
        gate = graph_gate_root / "analysis/decision.json"
        if read(gate).get("execution_passed") is not True:
            raise ValueError("actual graph/logit parity gate must pass first")
    parent = read(GATE / "plan.json")
    paths = [
        GATE / "plan.json",
        GATE / "analysis/decision.json",
        GATE / "inputs.json",
        PARENT / "ledger.json",
    ]
    references = {}
    for index in (0, 1):
        p = next((GATE / f"workers/{index}").glob("[0-9][0-9]-full.json"))
        row = read(p)
        if sha(Path(row["logits_path"])) != row["logits_sha256"]:
            raise ValueError("registered reference prediction artifact changed")
        references[str(index)] = row["logits_path"]
        paths += [p, Path(row["logits_path"])]
    for path, expected in parent["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError("prior validation input changed")
    paths += [Path(p) for p in parent["bound_files"]]
    policies = {
        name: dict(parent["component_policies"]["full"], inline_repair=inline)
        for name, inline in (("original_probe", False), ("inline", True))
    }
    if indexed_graph:
        graph_plan = read(graph_gate_root / "plan.json")
        if digest(graph_plan["parent"]["config"]) != digest(parent["parent"]["config"]) or digest(
            read(graph_gate_root / "inputs.json")
        ) != digest(read(GATE / "inputs.json")):
            raise ValueError("graph gate uses different model or input controls")
        policies = dict(
            original_probe=graph_plan["component_policies"]["full"],
            inline=graph_plan["component_policies"]["inline_indexed"],
        )
        runtime_fields = {"inline_repair", "graph_positional_search", "graph_cosine_topk"}
        if {k: v for k, v in policies["original_probe"].items() if k not in runtime_fields} != {
            k: v for k, v in policies["inline"].items() if k not in runtime_fields
        }:
            raise ValueError("paired methods differ in algorithm, budget or quality policy")
        paths += [gate, graph_gate_root / "plan.json", graph_gate_root / "inputs.json"]
    root.mkdir(parents=True)
    write_once(root / "inputs.json", read(GATE / "inputs.json"))
    with (root / "cpu.txt").open("x") as output:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_paired_inline_latency.py",
                "mosaickv/tests/unit/test_inline_first_step_repair.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json",
        dict(passed=result.returncode == 0, source_sha=source, log_sha=sha(root / "cpu.txt")),
    )
    if result.returncode:
        raise ValueError("paired-runtime CPU gate failed; no GPU submission")
    paths += [root / "inputs.json", root / "cpu.json"]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            parent=parent["parent"],
            options=parent["options"],
            policies=policies,
            references=references,
            bound_files={str(p): sha(p) for p in paths},
            warmups=5,
            repetitions=20,
            seed=20261107,
            bootstrap_replicates=10000,
            output_predictions=128,
            classification="two_consumed_input_randomized_paired_runtime_engineering_validation",
            original_source_codec_unchanged=True,
            optional_zero_row_codec_enabled=False,
            indexed_positional_graph_in_inline_arm=indexed_graph,
            source_codec_decision=(
                "retain validated default; optional zero-row codec remains "
                "a disclosed storage/latency tradeoff"
            ),
            fresh_evaluation_authorized=False,
            larger_campaign_authorized=False,
            all_baseline_superiority_claim_authorized=False,
        ),
    )
    import six_mechanism_feasibility as infra

    infra.SOURCE = SOURCE
    infra.SHELL = SOURCE / "mosaickv/slurm/benchmark_inline_repair.sbatch"
    infra.launch(root, "inline_pair")


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import integrated_storage
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("identical deterministic launch configuration required")
    torch.set_num_threads(12)
    torch.manual_seed(plan["parent"]["protocol"]["seed"])
    torch.use_deterministic_algorithms(True)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    cfg = RunConfig.from_mapping(plan["parent"]["config"])
    rows, errors = [], []
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            folder / "manifest.json",
            dict(
                git_sha=plan["source_sha"],
                source_dirty=False,
                config_sha=digest(plan),
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                environment=environment(),
                cuda=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                gpu_count=1,
                backend="huggingface",
                attention="eager",
                precision="bf16",
                seed=plan["parent"]["protocol"]["seed"],
                method_order_seed=plan["seed"] + index,
                measurement_type="systems_microbenchmark",
                **scheduler_identity({}),
                fixed_clocks_applied=False,
                reference_logits_sha256=sha(Path(plan["references"][str(index)])),
            ),
        )
        inputs = read(root / "inputs.json")[index]
        reference = torch.load(
            plan["references"][str(index)], weights_only=True, map_location="cpu"
        )
        transport = dict(plan["parent"], options=plan["options"])
        options = campaign.request_options(
            plan["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
        )
        rng = random.Random(plan["seed"] + index)
        with (folder / "raw.jsonl").open("x") as stream:
            for phase, count in (("warmup", plan["warmups"]), ("timed", plan["repetitions"])):
                orders = [
                    ("original_probe", "inline") if i % 2 == 0 else ("inline", "original_probe")
                    for i in range(count)
                ]
                rng.shuffle(orders)
                for trial, order in enumerate(orders):
                    pair = []
                    for arm in order:
                        row = dict(
                            method_id="evicastkv_storage_six_precision_overlay_v1",
                            sample_id=inputs["sample_id"],
                            phase=phase,
                            trial=trial,
                            arm=arm,
                            status="failed",
                            manifest_path=str(folder / "manifest.json"),
                            method_order=list(order),
                        )
                        try:
                            gc.collect()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            policy = OverlayPolicy(
                                **dict(
                                    plan["policies"][arm],
                                    layers=tuple(plan["policies"][arm]["layers"]),
                                )
                            )
                            started = perf_counter()
                            begin.record()
                            with integrated_storage(
                                adapter, policy, reader_options=options
                            ) as proof:
                                values, detail = campaign.execute(
                                    adapter, inputs, transport, "mosaickv_exact", audit=False
                                )
                            end.record()
                            end.synchronize()
                            elapsed = perf_counter() - started
                            row.update(
                                status="completed",
                                request_wall_seconds=elapsed,
                                request_cuda_interval_ms=begin.elapsed_time(end),
                                max_memory_allocated=torch.cuda.max_memory_allocated(),
                                max_memory_reserved=torch.cuda.max_memory_reserved(),
                                detail=detail,
                                mechanism_evidence=proof,
                                all_128_logits_bitwise=torch.equal(values.cpu(), reference),
                            )
                            if not row["all_128_logits_bitwise"]:
                                raise RuntimeError(
                                    "timing repetition changed registered 128-prediction output"
                                )
                            pair.append(row)
                            del values
                        except Exception:
                            row.update(status="failed", error=traceback.format_exc())
                            errors.append(row["error"])
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        rows.append(row)
                        if errors:
                            break
                    if not errors:
                        keys = (
                            "initial_encoded_bytes",
                            "final_encoded_bytes",
                            "reported_counter_peak_bytes",
                        )
                        if any(pair[0]["detail"][k] != pair[1]["detail"][k] for k in keys):
                            errors.append(
                                "paired policies do not match measured storage boundaries"
                            )
                        if (
                            pair[0]["mechanism_evidence"]["provisional_entropy"]
                            != pair[1]["mechanism_evidence"]["provisional_entropy"]
                        ):
                            errors.append("original and inline entropy differ numerically")
                    if errors:
                        break
                if errors:
                    break
        write_once(folder / "gpu_after.json", asdict(capture_gpu_environment(torch)))
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            errors=errors,
            row_count=len(rows),
            source_sha=plan["source_sha"],
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    from mosaickv.evaluation.paired_inline_latency import summarize, validate_rows

    plan = checked(root, expected)
    results, errors, rows = {}, {}, []
    for index in (0, 1):
        folder = root / f"workers/{index}"
        result = read(folder / "result.json") if (folder / "result.json").exists() else {}
        if result.get("status") != "completed":
            errors[str(index)] = result.get("errors", ["incomplete worker"])
        part = (
            [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            if (folder / "raw.jsonl").exists()
            else []
        )
        rows += part
        if len(part) != 2 * (plan["warmups"] + plan["repetitions"]) or any(
            r["status"] != "completed" for r in part
        ):
            errors[str(index)] = [*errors.get(str(index), []), "incomplete/failed trials"]
            continue
        try:
            timed = validate_rows(part, warmups=plan["warmups"], repetitions=plan["repetitions"])
        except ValueError as error:
            errors[str(index)] = [*errors.get(str(index), []), str(error)]
            continue
        results[str(index)] = summarize(
            *[[r["request_wall_seconds"] for r in timed[a]] for a in ("original_probe", "inline")],
            seed=plan["seed"] + index,
            replicates=plan["bootstrap_replicates"],
        )
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(out / "trials.json", rows)
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            results=results,
            errors=errors,
            engineering_parity_and_latency_gate=not errors
            and len(results) == 2
            and all(r["paired_latency_engineering_gate"] for r in results.values()),
            all_baseline_superiority_established=False,
            independent_quality_established=False,
            fresh_evaluation_authorized=False,
            larger_campaign_authorized=False,
        ),
    )
    lines = [
        "# Paired inline-repair runtime engineering validation",
        "",
        "Two consumed inputs; no all-baseline superiority claim.",
        "",
        "| Input | Original mean s | Inline mean s | Ratio | Adjusted upper ratio | Pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, r in results.items():
        lines.append(
            f"| {i} | {r['original']['mean']:.4f} | {r['inline']['mean']:.4f} | "
            f"{r['paired_mean_ratio']:.5f} | {r['adjusted_one_sided_upper_ratio']:.5f} | "
            f"{r['paired_latency_engineering_gate']} |"
        )
    lines += [
        "",
        "Five warmups and twenty timed paired trials per method/input; randomized balanced "
        "order. Initial/final/counter-peak bytes must match and every 128-logit output must "
        "be bitwise identical. FullKV is not a latency competitor.",
        "",
        str(errors),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--phase", default="inline_pair")
    parser.add_argument("--indexed-graph", action="store_true")
    parser.add_argument("--graph-gate-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.root, args.indexed_graph, args.graph_gate_root)
    if args.action == "worker":
        return worker(args.root, args.plan_sha)
    if args.action == "analyze":
        return analyze(args.root, args.plan_sha)
    checked(args.root, args.plan_sha)
    return subprocess.call(
        [
            PYTHON,
            str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
            "--policy",
            str(args.root / f"{args.phase}_admission.json"),
            "--state",
            str(args.root / f"{args.phase}_admission_state.json"),
            "--apply",
            "--watch",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
