#!/usr/bin/env python3
"""Finite CPU -> native controls -> consumed paired model verification."""

import argparse
import gc
import json
import os
import random
import subprocess
import traceback
import weakref
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
CAPTURES = Path("/scratch/djy8hg/runs/evicastkv/all-head-source-feasibility-20261107-v1")
MODEL = Path("/scratch/djy8hg/runs/evicastkv/component-admission-paired-20261107-v3")
ARMS = ("storage_only_common", "storage_only_complete_source_replacement")
BOUNDARIES = ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable complete-owner plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(k)) != v for k, v in p["bound_files"].items()):
        raise ValueError("consumed source archives or frozen controls changed")
    return p


def submit(root, action, after=None):
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=mk-owner-{action}",
    ]
    if action == "gpu":
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:45:00",
            "--array=0-1%2",
            "--hold",
        ]
    else:
        args += [
            "--partition=bii",
            "--cpus-per-task=4",
            "--mem=12G",
            "--time=01:00:00" if action == "controller" else "--time=00:20:00",
        ]
        if action == "cpu":
            args += ["--array=0-1%2"]
    if after:
        args += [f"--dependency=afterany:{after}"]
    args += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/complete_owner_study.sbatch"),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable complete-owner study root required")
    parent = read(MODEL / "plan.json")
    if read(MODEL / "analysis/decision.json")["complete"] is not True:
        raise ValueError("complete frozen storage-only execution controls required")
    paths = [
        MODEL / "plan.json",
        MODEL / "analysis/decision.json",
        CAPTURES / "plan.json",
        SOURCE / "mosaickv/docs/complete_owner_replacement.md",
    ]
    paths += [Path(k) for k in parent["bound_files"]]
    for i in (0, 1):
        paths += [
            CAPTURES / f"workers/{i}/{n}"
            for n in ("source_codes.pt", "source_codes.json", "manifest.json")
        ]
        paths.append(Path(parent["references"][str(i)]["storage"]))
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu-tests.txt").open("x") as out:
        test = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_predictive_source.py",
                "mosaickv/tests/unit/test_all_head_source_gpu.py",
                "mosaickv/tests/unit/test_all_head_source_codec.py",
                "mosaickv/tests/unit/test_layer_code_delta.py",
                "mosaickv/tests/unit/test_component_admission.py",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if test.returncode:
        raise ValueError("CPU regression failure; no Slurm submission")
    paths.append(root / "cpu-tests.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in paths},
            source_archive_root=str(CAPTURES),
            model_controls=parent,
            layers=32,
            samples=2,
            cpu_predictors=["xor1", "subtract1", "xor8", "subtract8", "delta_add"],
            GPU_predictor="delta_add_only",
            minimum_source_stream_plus_replay_saving_fraction=0.02,
            seed=20261107,
            warmups=5,
            repetitions=20,
            output_predictions=128,
            bootstrap_replicates=10000,
            multiplicity_comparisons=2,
            source_precision_and_original_positions_unchanged=True,
            benchmark_labels_accessed=False,
            fresh_or_paper_campaign_authorized=False,
            independent_quality_N=0,
            whole_request_peak_certified=False,
            measurement_type="complete_source_owner_CPU_then_consumed_model_engineering_not_quality",
        ),
    )
    array = submit(root, "cpu")
    write_once(
        root / "submission.json",
        dict(
            cpu_array=array,
            cpu_analysis_and_conditional_launcher=submit(root, "follow", array),
            GPUs_initially=0,
        ),
    )
    print(json.dumps(read(root / "submission.json")), flush=True)


def cpu(root, expected):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.evaluation.all_head_source_codec import SourceCoverage
    from mosaickv.evaluation.predictive_source import screen
    from mosaickv.evaluation.scheduler_identity import scheduler_identity

    p = checked(root, expected)
    torch.set_num_threads(1)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"cpu/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    origin = Path(p["source_archive_root"]) / f"workers/{i}"
    old = read(origin / "manifest.json")
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            config_sha=expected,
            source_dirty=False,
            model=old["model"],
            origin_manifest=old,
            dataset=dict(
                id="saved_consumed_source_code_fixture", revision=sha(origin / "source_codes.pt")
            ),
            environment=environment(),
            CUDA_build=torch.version.cuda,
            CUDA_runtime="not_used_CPU",
            driver="not_used",
            gpu_type="not_used",
            gpu_count=0,
            backend="CPU_exact_code_fixture",
            attention="not_used",
            vllm="not_used",
            sglang="not_used",
            seed=p["seed"],
            measurement_type=p["measurement_type"],
            **scheduler_identity({}),
        ),
    )
    rows, errors = [], []
    try:
        archive = torch.load(origin / "source_codes.pt", map_location="cpu", weights_only=True)
        if sorted(archive) != list(range(p["layers"])):
            raise ValueError("all original source layers required")
        for layer, current in sorted(archive.items()):
            start = perf_counter()
            rows_count = current.numel() * 8 // (3 * 4096)
            proof = SourceCoverage(
                layer, tuple(range(32)), ("key", "value"), tuple(range(rows_count)), ((0, 4096),)
            )
            coverage = proof.verify(kv_heads=32, head_dim=128, length=rows_count)
            _, evidence = screen(
                None if layer == 0 else archive[layer - 1].numpy(), current.numpy()
            )
            additive = next(r["bytes"] for r in evidence["proposals"] if r["mode"] == "delta_add")
            rows.append(
                dict(
                    layer=layer,
                    **evidence,
                    **coverage,
                    additive_or_common_bytes=min(evidence["common_bytes"], additive),
                    source_replay_bytes=current.numel(),
                    CPU_seconds=perf_counter() - start,
                )
            )
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(status="failed" if errors else "completed", errors=errors, rows=rows),
    )
    return int(bool(errors))


def follow(root, expected):
    p = checked(root, expected)
    errors, totals = [], {}
    for i in (0, 1):
        try:
            r = read(root / f"cpu/{i}/result.json")
            if (
                r["status"] != "completed"
                or r["errors"]
                or [v["layer"] for v in r["rows"]] != list(range(p["layers"]))
            ):
                raise ValueError(r)
            if not all(
                v["original_code_bits_exact"] and v["all_shared_consumers_covered"]
                for v in r["rows"]
            ):
                raise ValueError("complete exact source coverage failed")
            replay = max(v["source_replay_bytes"] for v in r["rows"])
            before = sum(v["common_bytes"] for v in r["rows"]) + replay
            after = sum(v["additive_or_common_bytes"] for v in r["rows"]) + replay
            totals[str(i)] = dict(
                parent_source_plus_replay_bytes=before,
                candidate_source_plus_replay_bytes=after,
                charged_replay_bytes=replay,
                saving_fraction=1 - after / before,
                additive_layers=sum(
                    v["additive_or_common_bytes"] < v["common_bytes"] for v in r["rows"]
                ),
            )
        except Exception:
            errors.append(traceback.format_exc())
    passed = (
        not errors
        and len(totals) == 2
        and all(
            v["saving_fraction"] >= p["minimum_source_stream_plus_replay_saving_fraction"]
            for v in totals.values()
        )
    )
    write_once(
        root / "cpu-analysis/decision.json",
        dict(
            complete=not errors,
            errors=errors,
            results=totals,
            CPU_source_gate_passed=passed,
            GPU_KV_or_quality_claim=False,
        ),
    )
    if passed:
        array = submit(root, "gpu")
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
        write_once(
            root / "gpu-submission.json",
            dict(
                array=array,
                controller=submit(root, "controller"),
                analyzer=submit(root, "analyze", array),
                maximum_GPUs=2,
                cpus_per_worker=12,
            ),
        )
    print(
        json.dumps(dict(CPU_source_gate_passed=passed, results=totals, errors=errors)), flush=True
    )


def operator_controls(p, index):
    import numpy as np
    import torch

    from mosaickv.evaluation import all_head_source_gpu as native
    from mosaickv.evaluation.predictive_source import encode as oracle

    archive = torch.load(
        Path(p["source_archive_root"]) / f"workers/{index}/source_codes.pt",
        map_location="cpu",
        weights_only=True,
    )
    previous, previous_cpu, records, references = None, None, [], []
    for layer in range(p["layers"]):
        current = archive[layer].cuda()
        packet, _selected, _legacy, parent_bytes = native.encode_best(
            previous, current, additive_candidate=True
        )
        selected = getattr(packet, "additive", False)
        if selected:
            expected = oracle(previous_cpu, archive[layer].numpy(), "delta_add").body
            if not np.array_equal(
                packet.payload.cpu().numpy(), expected.payload
            ) or not np.array_equal(packet.offsets.cpu().numpy().view(np.uint32), expected.offsets):
                raise ValueError("native additive packet differs from exact CPU bytes")
            restored = native.decode(previous, packet, implementation="bitmap_prefix")
            if not torch.equal(restored, current):
                raise ValueError("native source bit roundtrip failed")
            previous_snapshot = previous.clone()
            scratch = previous.clone()
            native.decode(scratch, packet, scratch, implementation="bitmap_prefix")
            if not torch.equal(scratch, current):
                raise ValueError("in-place source replay changed original codes")
            # Equal predecessor/current is legal, but is not a changed-source
            # independence witness by itself.
            changed = not torch.equal(previous, scratch)
            if not torch.equal(previous, previous_snapshot):
                raise ValueError("isolated replay mutated original predecessor")
            references.append(weakref.ref(current))
            del restored, scratch, previous_snapshot
        else:
            changed = False
        records.append(
            dict(
                layer=layer,
                additive_selected=selected,
                parent_bytes=parent_bytes,
                selected_bytes=packet.serialized_bytes + 1,
                native_source_bits_exact=True,
                isolated_in_place_replay_verified=selected,
                changed_source_control=changed,
            )
        )
        previous, previous_cpu = current, archive[layer].numpy()
    del previous, current, packet
    gc.collect()
    if not references or any(ref() is not None for ref in references):
        raise ValueError("accepted complete source owner was retained")
    torch.cuda.synchronize()
    return dict(
        passed=True,
        rows=records,
        retired_additive_source_owners=len(references),
        source_scope="every original feature and all 64 K/V consumers",
        quality_claim=False,
    )


def gpu(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.all_head_source_gpu import use as common_codec
    from mosaickv.evaluation.component_admission import component_scope
    from mosaickv.evaluation.predictive_source_gpu import use as replacement
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy
    from mosaickv.graph.structural_templates import engine_scope
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    p = checked(root, expected)
    if read(root / "cpu-analysis/decision.json")["CPU_source_gate_passed"] is not True:
        raise ValueError("positive complete CPU gate required before GPU")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUDA setup required before Python")
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"gpu/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    controls = p["model_controls"]
    features = read(controls["inputs"])[i]
    cfg = RunConfig.from_mapping(controls["parent"]["config"])
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            config_sha=expected,
            source_dirty=False,
            model=asdict(cfg.model),
            dataset=dict(
                id="COCO2014 captions",
                revision="c3213b54e173ae86517d481de17f31439d6f8c72",
                split="consumed_validation_timing_inputs",
            ),
            sample_id=features["sample_id"],
            environment=environment(),
            cuda=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            gpu_count=1,
            backend="HF_eager",
            attention_implementation="eager",
            precision="bf16",
            seed=p["seed"],
            vllm="not_used",
            sglang="not_used",
            measurement_type=p["measurement_type"],
            **scheduler_identity({}),
        ),
    )
    errors, rows = [], 0
    try:
        write_once(folder / "operator-controls.json", operator_controls(p, i))
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        transport = dict(controls["parent"], options=controls["options"])
        options = campaign.request_options(
            controls["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
        )
        reference = torch.load(
            controls["references"][str(i)]["storage"], map_location="cpu", weights_only=True
        )
        policy = OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
        rng = random.Random(p["seed"] + i)
        with engine_scope(), (folder / "raw.jsonl").open("x") as out:
            for phase, count in (("warmup", p["warmups"]), ("timed", p["repetitions"])):
                orders = [list(ARMS[t % 2 :] + ARMS[: t % 2]) for t in range(count)]
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
                            begin = perf_counter()
                            codec_scope = (
                                replacement()
                                if arm == ARMS[1]
                                else common_codec(
                                    True, construction="pack_selected", decoder="bitmap_prefix"
                                )
                            )
                            with (
                                codec_scope as codec,
                                component_scope(
                                    adapter,
                                    policy,
                                    mode="storage_preservation",
                                    reader_options=options,
                                ) as evidence,
                            ):
                                logits, detail = campaign.execute(
                                    adapter, features, transport, "mosaickv_exact", audit=False
                                )
                            torch.cuda.synchronize()
                            elapsed = perf_counter() - begin
                            actual = logits.cpu()
                            if not torch.equal(actual, reference):
                                raise ValueError("128 frozen parent vocabulary vectors changed")
                            proof = codec.report()
                            if (
                                not proof["all_shared_source_consumers_covered"]
                                or len(proof["records"]) != 32
                            ):
                                raise ValueError("complete source consumer coverage required")
                            if arm == ARMS[1] and (
                                not proof["additive_replacements"]
                                or proof["additive_original_owners_retired"]
                                != proof["additive_replacements"]
                            ):
                                raise ValueError(
                                    "new source was not selected or old source owners remain alive"
                                )
                            if any(
                                type(detail[k]) is not int or detail[k] <= 0 for k in BOUNDARIES
                            ):
                                raise ValueError("complete physical storage boundaries required")
                            if evidence["extra_gpu_bytes"] or evidence["residual_cpu_bytes"]:
                                raise ValueError(
                                    "storage-only path unexpectedly created overlay storage"
                                )
                            row.update(
                                status="completed",
                                request_wall_seconds=elapsed,
                                all_128_frozen_logits_bitwise=True,
                                codec=proof,
                                detail=detail,
                                mechanism_evidence=evidence,
                            )
                            if phase == "warmup" and trial == 0:
                                torch.save(actual, folder / f"{arm}.logits.pt")
                            del logits, actual
                        except Exception:
                            row["error"] = traceback.format_exc()
                            errors.append(row["error"])
                        out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        out.flush()
                        rows += 1
                        if errors:
                            raise RuntimeError("stop on first failed correctness trial")
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(status="failed" if errors else "completed", errors=errors, raw_rows=rows),
    )
    return int(bool(errors))


def analyze(root, expected):
    from mosaickv.evaluation.source_hybrid_validation import paired_summary

    p = checked(root, expected)
    errors, results = [], {}
    for i in (0, 1):
        try:
            folder = root / f"gpu/{i}"
            if read(folder / "result.json")["status"] != "completed":
                raise ValueError(read(folder / "result.json"))
            rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            keyed = {(r["phase"], r["trial"], r["arm"]): r for r in rows}
            wanted = {
                (phase, t, a)
                for phase, n in (("warmup", p["warmups"]), ("timed", p["repetitions"]))
                for t in range(n)
                for a in ARMS
            }
            if len(keyed) != len(rows) or set(keyed) != wanted:
                raise ValueError("all unique paired warmup and timed rows required")
            if not all(
                r["status"] == "completed" and r["all_128_frozen_logits_bitwise"] for r in rows
            ):
                raise ValueError("failed or numerically invalid model row")
            for phase, trial, _a in wanted:
                if any(
                    keyed[phase, trial, ARMS[1]]["detail"][k]
                    >= keyed[phase, trial, ARMS[0]]["detail"][k]
                    for k in BOUNDARIES
                ):
                    raise ValueError(
                        "replacement did not lower every measured decode-owned storage boundary"
                    )
            timed = {a: [keyed["timed", t, a] for t in range(p["repetitions"])] for a in ARMS}
            results[str(i)] = dict(
                sample_id=rows[0]["sample_id"],
                storage={
                    a: {k: sorted({r["detail"][k] for r in timed[a]}) for k in BOUNDARIES}
                    for a in ARMS
                },
                latency=paired_summary(
                    [r["request_wall_seconds"] for r in timed[ARMS[0]]],
                    [r["request_wall_seconds"] for r in timed[ARMS[1]]],
                    seed=p["seed"] + i,
                    replicates=p["bootstrap_replicates"],
                    comparisons=p["multiplicity_comparisons"],
                ),
                bitwise_parent_preservation=True,
            )
        except Exception:
            errors.append(traceback.format_exc())
    passed = (
        not errors
        and len(results) == 2
        and all(r["latency"]["paired_latency_engineering_gate"] for r in results.values())
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not errors,
            errors=errors,
            results=results,
            engineering_memory_and_latency_gate_passed=passed,
            new_quality_or_all_baseline_superiority_established=False,
            whole_request_peak_certified=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    lines = [
        "# Complete-source replacement engineering result",
        "",
        "Two consumed COCO inputs, LLaVA-1.5-7B BF16 eager, 128 predictions; "
        "five warmups and 20 paired trials.",
        "All reported peaks below are decode-owned, not certified whole-request KV peak.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Decode-owned peak MiB | Mean request s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, r in results.items():
        for a in ARMS:
            memory = [max(r["storage"][a][k]) / 2**20 for k in BOUNDARIES]
            seconds = r["latency"]["original" if a == ARMS[0] else "inline"]["mean"]
            lines.append(
                f"| {i} | {a} | {memory[0]:.4f} | {memory[1]:.4f} | "
                f"{memory[2]:.4f} | {seconds:.4f} |"
            )
        lines += [
            "",
            f"Input {i}: new/parent mean latency ratio "
            f"{r['latency']['paired_mean_ratio']:.6f}; adjusted upper "
            f"{r['latency']['adjusted_one_sided_upper_ratio']:.6f}.",
        ]
    lines += [
        "",
        f"Complete: {not errors}. Joint engineering gate: {passed}.",
        "This is common exact source coding, not a six-component quality result "
        "or independent benchmark evidence.",
        "Next: inspect the dominant measured cost if latency fails; do not relax "
        "parity/storage controls or launch a fresh campaign.",
    ]
    with (root / "analysis/report.md").open("x") as out:
        out.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(complete=not errors, engineering_gate=passed, results=results, errors=errors)
        ),
        flush=True,
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("prepare", "cpu", "follow", "gpu", "analyze", "controller"))
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--plan-sha")
    args = a.parse_args()
    if args.action == "prepare":
        return prepare(args.root)
    if args.action == "controller":
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
    return {"cpu": cpu, "follow": follow, "gpu": gpu, "analyze": analyze}[args.action](
        args.root, args.plan_sha
    )


if __name__ == "__main__":
    raise SystemExit(main())
