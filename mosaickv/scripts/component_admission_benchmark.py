#!/usr/bin/env python3
"""Bounded consumed-input test of explicitly reduced-component execution."""

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

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/direct-structural-key-GPU-20261107-v3-feasible-groups"
)
ARMS = ("common_storage_wrapped", "common_storage_hook_free", "retained_hybrid")
BOUNDARIES = ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("immutable package boundary required")
    if sha(root / "plan.json") != expected:
        raise ValueError("registered component admission plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if any(sha(Path(k)) != v for k, v in plan["bound_files"].items()):
        raise ValueError("bound controls or consumed references changed")
    return plan


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable admission benchmark root required")
    parent = read(PARENT / "plan.json")
    if read(PARENT / "analysis/decision.json")["complete"] is not True:
        raise ValueError("complete paired retained-path controls required")
    paths = [PARENT / "plan.json", PARENT / "analysis/decision.json"]
    paths += [Path(k) for k in parent["bound_files"]]
    policy = parent["policies"]["hybrid_reference"]
    if policy["graph_direct_metadata_key"] or policy["anchor_counterfactual"] != "full_heads":
        raise ValueError("retained corrected hybrid reference required")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as out:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_component_admission.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
                "mosaickv/tests/unit/test_layer_code_delta.py",
                "mosaickv/tests/unit/test_all_head_source_gpu.py",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise ValueError("CPU controls failed; no GPU submission")
    paths.append(root / "cpu.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            retained_implementation_sha=parent["source_sha"],
            bound_files={str(p): sha(p) for p in paths},
            arms=list(ARMS),
            parent=parent["parent"],
            options=parent["options"],
            inputs=parent["inputs"],
            references=parent["references"],
            retained_policy=policy,
            seed=20261107,
            output_predictions=128,
            warmups=6,
            repetitions=24,
            bootstrap_replicates=10000,
            multiplicity_comparisons=6,
            common_codec_identical_all_arms=True,
            hook_free_is_explicit_source_only_not_six_component_method=True,
            comparison="component_elimination_engineering_not_byte_matched_selection_policy",
            independent_quality_N=0,
            fresh_or_paper_campaign_authorized=False,
            whole_request_peak_certified=False,
        ),
    )
    import all_head_native_source as infra

    infra.SOURCE, infra.SHELL = (
        SOURCE,
        SOURCE / "mosaickv/slurm/component_admission_benchmark.sbatch",
    )
    array = infra.submit(root, "worker")
    write_once(
        root / "admission.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            adopt_existing_holds=[array],
        ),
    )
    receipt = dict(
        array=array,
        controller=infra.submit(root, "controller"),
        analyzer=infra.submit(root, "analyze", after=array),
        maximum_GPU_workers=2,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.all_head_source_gpu import use as source_codec
    from mosaickv.evaluation.component_admission import component_scope
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy
    from mosaickv.graph.structural_templates import engine_scope
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    p = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("registered deterministic CUDA launch required")
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    features = read(p["inputs"])[i]
    if not features["sample_id"].startswith("coco_caption:"):
        raise ValueError("registered consumed COCO timing inputs required")
    cfg = RunConfig.from_mapping(p["parent"]["config"])
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            source_dirty=False,
            config_sha=digest(p),
            model=asdict(cfg.model),
            dataset=dict(
                id="COCO2014 captions",
                revision="c3213b54e173ae86517d481de17f31439d6f8c72",
                split="consumed_validation_timing_inputs",
            ),
            sample_id=features["sample_id"],
            historical_transport_config_dataset_not_actual_input_dataset=asdict(cfg.dataset),
            environment=environment(),
            CUDA=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            gpu_count=1,
            backend="huggingface",
            attention_implementation="eager",
            precision="bf16",
            vllm="not_used",
            sglang="not_used",
            seed=p["seed"],
            measurement_type="paired_consumed_component_elimination_not_independent_quality",
            cpu_torch_threads=1,
            fixed_clocks_applied=False,
            **scheduler_identity({}),
        ),
    )
    errors, count = [], 0
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        transport = dict(p["parent"], options=p["options"])
        options = campaign.request_options(
            p["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
        )
        refs = {
            k: torch.load(v, map_location="cpu", weights_only=True)
            for k, v in p["references"][str(i)].items()
        }
        source_policy = OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
        hybrid = OverlayPolicy(
            **dict(p["retained_policy"], layers=tuple(p["retained_policy"]["layers"]))
        )
        rng = random.Random(p["seed"] + i)
        with engine_scope(), (folder / "raw.jsonl").open("x") as log:
            for phase, repetitions in (("warmup", p["warmups"]), ("timed", p["repetitions"])):
                orders = [list(ARMS[t % 3 :] + ARMS[: t % 3]) for t in range(repetitions)]
                rng.shuffle(orders)
                for trial, order in enumerate(orders):
                    for arm in order:
                        row = dict(
                            sample_id=features["sample_id"],
                            phase=phase,
                            trial=trial,
                            arm=arm,
                            method_order=order,
                            status="failed",
                            manifest_path=str(folder / "manifest.json"),
                        )
                        try:
                            gc.collect()
                            torch.cuda.synchronize()
                            start = perf_counter()
                            with (
                                source_codec(
                                    True, construction="pack_selected", decoder="bitmap_prefix"
                                ) as codec,
                                component_scope(
                                    adapter,
                                    hybrid if arm == "retained_hybrid" else source_policy,
                                    mode="storage_preservation"
                                    if arm == "common_storage_hook_free"
                                    else "experimental_overlay",
                                    reader_options=options,
                                ) as evidence,
                            ):
                                logits, detail = campaign.execute(
                                    adapter, features, transport, "mosaickv_exact", audit=False
                                )
                            torch.cuda.synchronize()
                            elapsed = perf_counter() - start
                            wanted = refs["hybrid" if arm == "retained_hybrid" else "storage"]
                            actual = logits.cpu()
                            if not torch.equal(actual, wanted):
                                raise ValueError("128 frozen vocabulary vectors changed")
                            proof = codec.report()
                            if (
                                not proof["all_shared_source_consumers_covered"]
                                or len(proof["records"]) != 32
                            ):
                                raise ValueError("common codec did not cover every source consumer")
                            if arm == "common_storage_hook_free" and (
                                evidence["original_six_mechanism_design_active"] is not False
                                or evidence["extra_gpu_bytes"] != 0
                                or evidence["residual_cpu_bytes"] != 0
                                or evidence["repair_events"]
                            ):
                                raise ValueError("hook-free variant silently enabled an overlay")
                            if any(
                                type(detail[k]) is not int or detail[k] <= 0 for k in BOUNDARIES
                            ):
                                raise ValueError("complete physical storage boundaries required")
                            row.update(
                                status="completed",
                                request_wall_seconds=elapsed,
                                detail=detail,
                                codec=proof,
                                mechanism_evidence=evidence,
                                all_128_frozen_logits_bitwise=True,
                                greedy_parent_agreement=float(
                                    (actual.argmax(-1) == refs["storage"].argmax(-1)).float().mean()
                                ),
                            )
                            if phase == "warmup" and trial == 0:
                                torch.save(actual, folder / f"{arm}.logits.pt")
                            del logits, actual
                        except Exception:
                            row["error"] = traceback.format_exc()
                            errors.append(row["error"])
                        log.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                        log.flush()
                        count += 1
                        if errors:
                            raise RuntimeError(
                                "failed correctness trial; no further timing expansion"
                            )
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(status="failed" if errors else "completed", errors=errors, raw_rows=count),
    )
    return int(bool(errors))


def analyze(root, expected):
    from mosaickv.evaluation.source_hybrid_validation import paired_summary

    p = checked(root, expected)
    errors, results = [], {}
    pairs = (
        ("common_storage_wrapped", "common_storage_hook_free"),
        ("retained_hybrid", "common_storage_hook_free"),
        ("common_storage_wrapped", "retained_hybrid"),
    )
    for i in (0, 1):
        try:
            folder = root / f"workers/{i}"
            outcome = read(folder / "result.json")
            if outcome["status"] != "completed":
                raise ValueError(outcome)
            rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            keyed = {(r["phase"], r["trial"], r["arm"]): r for r in rows}
            planned = {
                (phase, trial, arm)
                for phase, n in (("warmup", p["warmups"]), ("timed", p["repetitions"]))
                for trial in range(n)
                for arm in ARMS
            }
            if len(keyed) != len(rows) or set(keyed) != planned:
                raise ValueError("complete unique warmup and timed trials required")
            for phase, trial, arm in planned:
                row = keyed[phase, trial, arm]
                if row["status"] != "completed" or not row["all_128_frozen_logits_bitwise"]:
                    raise ValueError("failed or numerically invalid row")
                for k in BOUNDARIES:
                    if (
                        keyed[phase, trial, "common_storage_wrapped"]["detail"][k]
                        != keyed[phase, trial, "common_storage_hook_free"]["detail"][k]
                    ):
                        raise ValueError("hook elimination changed live storage accounting")
            timed = {a: [keyed["timed", t, a] for t in range(p["repetitions"])] for a in ARMS}
            results[str(i)] = dict(
                sample_id=rows[0]["sample_id"],
                storage={
                    a: {k: sorted({r["detail"][k] for r in v}) for k in BOUNDARIES}
                    for a, v in timed.items()
                },
                comparisons={
                    f"{b}_versus_{a}": paired_summary(
                        [r["request_wall_seconds"] for r in timed[a]],
                        [r["request_wall_seconds"] for r in timed[b]],
                        seed=p["seed"] + i,
                        replicates=p["bootstrap_replicates"],
                        comparisons=p["multiplicity_comparisons"],
                    )
                    for a, b in pairs
                },
                all_logits_bitwise=True,
                all_greedy_parent_agreement=all(r["greedy_parent_agreement"] == 1.0 for r in rows),
            )
        except Exception:
            errors.append(traceback.format_exc())
    engineering_pass = not errors and all(
        result["all_greedy_parent_agreement"]
        and result["comparisons"]["common_storage_hook_free_versus_retained_hybrid"][
            "paired_latency_engineering_gate"
        ]
        for result in results.values()
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not errors,
            errors=errors,
            results=results,
            source_sha=p["source_sha"],
            reduced_component_variant_not_six_mechanism_superiority=True,
            source_only_component_elimination_engineering_gate=engineering_pass,
            new_independent_quality_or_all_baseline_superiority_established=False,
            whole_request_peak_certified=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    lines = [
        "# Component-admission engineering result",
        "",
        "Two consumed COCO timing inputs; six warmups and 24 paired trials per input. "
        "This is not independent benchmark evidence or a byte-matched selection comparison. "
        "All arms use the same source codec. The hook-free arm explicitly disables all six "
        "research overlays; the complete implementation remains available.",
        "",
        f"Execution complete: {not errors}. Source-only versus hybrid engineering gate: "
        f"{engineering_pass}.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Decode-owned peak MiB | Mean request s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, result in results.items():
        for arm in ARMS:
            pair = result["comparisons"][
                "common_storage_hook_free_versus_common_storage_wrapped"
                if arm != "retained_hybrid"
                else "retained_hybrid_versus_common_storage_wrapped"
            ]
            seconds = pair["original" if arm == "common_storage_wrapped" else "inline"]["mean"]
            storage = result["storage"][arm]
            values = [max(storage[k]) / 2**20 for k in BOUNDARIES]
            lines.append(
                f"| {i} | {arm} | {values[0]:.4f} | {values[1]:.4f} | "
                f"{values[2]:.4f} | {seconds:.4f} |"
            )
        lines += ["", f"Input {i} paired ratios (new/old):", ""]
        for name, comparison in result["comparisons"].items():
            lines.append(
                f"- {name}: {comparison['paired_mean_ratio']:.6f}; adjusted one-sided "
                f"upper {comparison['adjusted_one_sided_upper_ratio']:.6f}."
            )
    lines += [
        "",
        "Every completed trial matched all 128 frozen vocabulary vectors. "
        "This establishes numerical preservation on these consumed inputs, not "
        "higher CIDEr or superiority over every baseline. Initial/final/peak equality "
        "is required between source-only arms. Whole-request KV peak is not certified. "
        "A smaller hybrid is not fabricated: its additional overlay storage is reported.",
    ]
    if errors:
        lines += [
            "",
            "Failures are preserved in decision.json; no performance promotion is authorized.",
        ]
    with (root / "analysis/report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(dict(complete=not errors, results=results, errors=errors)), flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--plan-sha")
    args = a.parse_args()
    if args.action == "prepare":
        return prepare(args.root)
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
            str(args.root / "admission.json"),
            "--state",
            str(args.root / "admission_state.json"),
            "--apply",
            "--watch",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
