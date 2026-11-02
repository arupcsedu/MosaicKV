#!/usr/bin/env python3
"""Finite exact-geometry diagnostic and conditional same-input request benchmark."""

import argparse
import json
import os
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import evicastkv_code_delta_gpu as backend
import evicastkv_common_streaming as common
import numpy as np

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.source_row_geometry import GEOMETRIES

SOURCE = common.SOURCE
ASYNC = Path("/scratch/djy8hg/runs/evicastkv/async-text-bitwise-consumed-20261102-v1")
PARENT = ASYNC / "timing"


def defer(root, revision):
    if clean_source(SOURCE) != revision:
        raise ValueError("immutable geometry source required")
    decision = common.previous.read(ASYNC / "analysis/decision.json")
    if (
        not decision.get("complete")
        or decision.get("failures")
        or decision.get("constructor_memory_gate") is not True
    ):
        write_once(
            root / "blocked.json",
            dict(reason="preceding exact-text constructor failed", GPU_submitted=False),
        )
        return
    dependency = common.previous.read(PARENT / "submission.json")["cpu_analysis"]
    control = root.with_name(root.name + "-launch")
    job = common.control.parent.subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:25:00",
            f"--dependency=afterany:{dependency}",
            "--job-name=mk-row-geometry-start",
            f"--output={control}/start-%j.out",
            f"--error={control}/start-%j.err",
            str(SOURCE / "mosaickv/slurm/evicastkv_row_geometry.sbatch"),
            str(SOURCE),
            str(root),
            revision,
            "start",
        ],
        text=True,
    ).strip()
    write_once(
        control / "deferred.json",
        dict(
            job_id=job,
            dependency=dependency,
            source_sha=revision,
            parent_plan_sha256=sha(PARENT / "plan.json"),
            parent_submission_sha256=sha(PARENT / "submission.json"),
        ),
    )


def start(root, revision):
    if clean_source(SOURCE) != revision or root.exists():
        raise ValueError("clean geometry source and new root required")
    parent = common.previous.read(PARENT / "plan.json")
    decision = common.previous.read(PARENT / "analysis/decision.json")
    if (
        not decision.get("complete")
        or decision.get("failures")
        or decision.get("plan_sha256") != sha(PARENT / "plan.json")
    ):
        raise ValueError("complete preceding request-timing evidence required")
    if decision.get("all_method_similar_time_gate") is True:
        write_once(
            root / "not_needed.json",
            dict(
                reason="preceding all-method engineering timing gate passed",
                GPU_submitted=False,
                paper_superiority_established=False,
            ),
        )
        return
    bound = dict(parent["bound_files"])
    for p in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        ASYNC / "analysis/decision.json",
        SOURCE / "mosaickv/docs/source_row_geometry_20261102.md",
    ):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            parent,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_source_row_geometry_diagnostic",
            paired_common_streaming_timing=False,
            maximum_worker_minutes=30,
            maximum_simultaneous_workers=2,
            geometry_arms=GEOMETRIES,
            geometry_warmups=5,
            geometry_repetitions=20,
            no_additional_memory_or_model_changes=True,
        ),
    )
    expected = sha(root / "plan.json")
    common.previous.cpu(root, expected)
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        return original(
            [
                str(SOURCE / "mosaickv/slurm/evicastkv_row_geometry.sbatch")
                if str(x).endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ],
            **kwargs,
        )

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation import shared_source_format
    from mosaickv.evaluation.layer_code_delta import convert
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.sequential_kv_lease import fill_component
    from mosaickv.evaluation.shared_source_kernel import project
    from mosaickv.evaluation.source_row_geometry import geometry
    from mosaickv.evaluation.text_source_storage import TextSourceFormat
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one deterministic isolated A100 required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(common.previous.resolved_mapping(plan, job))
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = common.previous.read(
            common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json"
        )
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            config_sha256=digest(dict(plan=expected, index=index)),
            plan_sha256=expected,
            run_id=str(out),
            measurement_type="diagnostic_measured",
            environment=common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            independent_quality_N=0,
            evaluator_commit=plan["source_sha"],
        )
        write_once(out / "manifest.json", manifest)
        messages = common.previous.load_messages(common.previous.read(job["inputs"][0]))
        profile = tuple(common.previous.read(Path(plan["profile_path"])))
        with patch.object(
            shared_source_format,
            "fit_transport",
            partial(shared_source_format.fit_transport, componentwise=True),
        ):
            runtime, packets, memory = backend.construct(
                adapter, messages, profile, "mosaickv_exact", audit=True
            )
        reference_folder = common.AUDIT / f"workers/{job['reference_worker']}"
        frozen = common.previous.read(reference_folder / "offline_payload.json")
        if (
            common.control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
            != frozen["payload_sha256"]
            or memory["final_prefill_cache_bytes"] != frozen["physical_bytes"]
        ):
            raise RuntimeError("geometry diagnostic changed original constructor")
        write_once(out / "constructor_memory.json", memory)
        expected_logits = torch.load(
            reference_folder / "greedy_logits.pt", weights_only=True, map_location="cpu"
        )
        variants = {}
        records = []
        for rows, heads in GEOMETRIES:
            key = f"r{rows}_h{heads}"
            record = dict(
                geometry=[rows, heads],
                status="failed",
                layers_checked=0,
                errors=[],
                resources=[],
                manifest_path=str(out / "manifest.json"),
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                independent_quality_N=0,
                method="mosaickv_exact",
                variant="source_precision_geometry_diagnostic",
                sample_id=common.previous.read(job["inputs"][0])["sample_id"],
                task=common.previous.read(job["inputs"][0])["task"],
                run_id=str(out),
            )
            try:
                candidate = tuple(geometry(p, rows, heads) for p in packets)
                if (
                    backing_bytes(tuple(t for p in candidate for t in p.tensors()))
                    != frozen["physical_bytes"]
                ):
                    raise RuntimeError("geometry changed an active allocation")
                with torch.inference_mode():
                    for old, new, block in zip(
                        packets, candidate, adapter._language_layers(), strict=True
                    ):
                        left, right = (
                            old.materialize(block.self_attn),
                            new.materialize(block.self_attn),
                        )
                        if not all(
                            torch.equal(a.view(torch.int16), b.view(torch.int16))
                            for a, b in zip(left, right, strict=True)
                        ):
                            raise RuntimeError("paired reconstruction changed a K/V bit")
                        del left, right
                        template = new.base if isinstance(new, TextSourceFormat) else new
                        a = torch.empty(new.shape, device="cuda", dtype=torch.bfloat16)
                        b = torch.empty_like(a)
                        for component in (0, 1):
                            fill_component(old, block.self_attn, component, a)
                            fill_component(new, block.self_attn, component, b)
                            if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                                raise RuntimeError("sequential reconstruction changed a K/V bit")
                            resource = project(
                                template,
                                block.self_attn,
                                (b, b),
                                component=component,
                                inspect_kernel=True,
                            )
                            if type(resource.get("spills")) is not int:
                                raise RuntimeError("kernel spill metadata unavailable")
                            record["resources"].append(resource)
                        del a, b
                        record["layers_checked"] += 1
                converted, codec = convert(candidate)
                if (
                    not codec["enabled"]
                    or codec["net_active_bytes"]
                    != plan["optimized_source_initial_bytes"][str(job["input_index"])]
                ):
                    raise RuntimeError("geometry changed encoded storage or replay scratch")
                values, detail = backend.decode_run(adapter, runtime, converted, audit=False)
                if not torch.equal(values, expected_logits):
                    raise RuntimeError("geometry changed a frozen 128-token vocabulary logit")
                record.update(
                    status="completed",
                    generation_128_logits_bitwise=True,
                    initial_bytes=codec["net_active_bytes"],
                    storage=detail["storage"],
                    max_spills=max(r["spills"] for r in record["resources"]),
                    model_probe_not_performance_evidence=True,
                )
                variants[key] = candidate
                del values, converted
            except Exception as exc:
                record["errors"].append(f"{type(exc).__name__}: {exc}")
                if key == "r16_h4":
                    raise
            records.append(record)
            write_once(out / (key + ".json"), record)
        rng = random.Random(plan["seed"] + index)
        events = []
        with torch.inference_mode():
            for layer, block in enumerate(adapter._language_layers()):
                target = torch.empty(packets[layer].shape, device="cuda", dtype=torch.bfloat16)
                for trial in range(-5, 20):
                    order = list(variants)
                    rng.shuffle(order)
                    for key in order:
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        for component in (0, 1):
                            fill_component(variants[key][layer], block.self_attn, component, target)
                        end.record()
                        events.append((key, layer, trial, start, end))
                torch.cuda.synchronize()
                del target
        torch.cuda.synchronize()
        trials = [
            dict(
                geometry=k,
                layer=layer_index,
                trial=t,
                warmup=t < 0,
                cuda_stream_ms=a.elapsed_time(b),
                sample_id=common.previous.read(job["inputs"][0])["sample_id"],
                manifest_path=str(out / "manifest.json"),
                measurement_type="diagnostic_measured",
                not_complete_request_latency=True,
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                method="mosaickv_exact",
            )
            for k, layer_index, t, a, b in events
        ]
        with (out / "trials.jsonl").open("x") as stream:
            for row in trials:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(
            out / "geometry.json",
            dict(
                records=records,
                constructor_memory_boundary=memory["boundary"],
                whole_request_gpu_peak_certified=False,
                original_packets_alive_only_in_this_diagnostic=True,
                independent_quality_N=0,
            ),
        )
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def select_geometry(records, trials, seed):
    """One common bit-identical geometry; >=5% fill gain on BOTH consumed inputs."""
    candidates = []
    summaries = []
    for rows, heads in GEOMETRIES[1:]:
        key = f"r{rows}_h{heads}"
        bounds = []
        eligible = True
        for i in (0, 1):
            group = records[i]
            baseline = next(r for r in group if r["geometry"] == [16, 4])
            candidate = next(r for r in group if r["geometry"] == [rows, heads])
            valid = (
                candidate["status"] == "completed"
                and candidate.get("layers_checked") == 32
                and candidate.get("generation_128_logits_bitwise") is True
                and candidate.get("max_spills", float("inf")) <= baseline["max_spills"]
            )
            if not valid:
                eligible = False
                continue
            arrays = {
                name: np.array(
                    [
                        sum(
                            r["cuda_stream_ms"]
                            for r in trials[i]
                            if r["geometry"] == name and r["trial"] == t
                        )
                        for t in range(20)
                    ]
                )
                for name in ("r16_h4", key)
            }
            if any(not np.isfinite(a).all() or not (a > 0).all() for a in arrays.values()):
                raise ValueError("finite complete geometry trials required")
            ids = np.random.default_rng(seed + i).integers(0, 20, size=(10000, 20))
            ratios = arrays[key][ids].mean(1) / arrays["r16_h4"][ids].mean(1)
            upper = float(np.quantile(ratios, 1 - 0.05 / (2 * 6)))
            bounds.append(upper)
            summaries.append(
                dict(
                    input_index=i,
                    geometry=[rows, heads],
                    mean_ratio=float(arrays[key].mean() / arrays["r16_h4"].mean()),
                    familywise_upper_ratio=upper,
                    pairing="same consumed input and trial",
                    not_task_quality_evidence=True,
                )
            )
        if eligible and len(bounds) == 2 and max(bounds) <= 0.95:
            candidates.append((max(bounds), rows, heads))
    return (list(min(candidates)[1:]) if candidates else None), summaries


def analyze(root, expected):
    plan = common.previous.common.inputs(root, expected)
    records = {}
    trials = {}
    failures = []
    for i in (0, 1):
        folder = root / f"workers/{i}"
        try:
            result = common.previous.read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
            ):
                raise ValueError("incomplete producer")
            if any(sha(folder / n) != h for n, h in result["outputs"].items()):
                raise ValueError("changed producer output")
            records[i] = common.previous.read(folder / "geometry.json")["records"]
            trials[i] = [
                json.loads(line) for line in (folder / "trials.jsonl").read_text().splitlines()
            ]
            if len(records[i]) != 4 or {tuple(r["geometry"]) for r in records[i]} != set(
                GEOMETRIES
            ):
                raise ValueError("all four distinct geometry outcomes required")
            for r in records[i]:
                if r["status"] == "completed":
                    group = [
                        t
                        for t in trials[i]
                        if t["geometry"] == f"r{r['geometry'][0]}_h{r['geometry'][1]}"
                    ]
                    keys = {(t["layer"], t["trial"]) for t in group}
                    if len(group) != 32 * 25 or keys != {
                        (layer, trial) for layer in range(32) for trial in range(-5, 20)
                    }:
                        raise ValueError("incomplete geometry repetition grid")
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{i}: {exc}")
    selected, summaries = (None, []) if failures else select_geometry(records, trials, plan["seed"])
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            selected_geometry=selected,
            summaries=summaries,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            paper_superiority_established=False,
        ),
    )
    if selected is None:
        return
    target = root / "timing"
    parent = common.previous.read(PARENT / "plan.json")
    bound = dict(plan["bound_files"])
    bound[str(root / "analysis/decision.json")] = sha(root / "analysis/decision.json")
    bound[str(root / "plan.json")] = expected
    write_once(
        target / "plan.json",
        dict(
            parent,
            source_sha=plan["source_sha"],
            bound_files=bound,
            source_geometry=selected,
            geometry_KV_and_128_logits_bitwise=True,
            stage="consumed_geometry_complete_request_timing",
        ),
    )
    for name in ("cpu.json", "cpu.xml"):
        os.link(root / name, target / name)
    common.submit(target, sha(target / "plan.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("defer", "start", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode in ("start", "defer"):
        globals()[args.mode](args.root, args.source_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)
