#!/usr/bin/env python3
"""Compare all-head codec/hybrid factorial against frozen storage-only baseline."""

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

SOURCE = Path(__file__).resolve().parents[2]
NATIVE = Path("/scratch/djy8hg/runs/evicastkv/all-head-native-source-20261107-v1")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/live-anchor-risk-repair-20261107-v1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(p):
    return json.loads(Path(p).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable factorial plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for p, checksum in plan["bound_files"].items():
        if sha(Path(p)) != checksum:
            raise ValueError("immutable source/comparison controls changed")
    return plan


def launch(root):
    import all_head_native_source as infra

    infra.SOURCE = SOURCE
    infra.SHELL = SOURCE / "mosaickv/slurm/all_head_source_model.sbatch"
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
        gpu_count_per_worker=1,
        cpus_per_worker=12,
        maximum_gpu_workers=2,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def prepare(root):
    import inspect

    from mosaickv.evaluation.common_streaming_timing import request

    source = clean_source(SOURCE)
    if not read(NATIVE / "analysis/decision.json")["passed"]:
        raise ValueError("native original-code parity required first")
    if root.exists():
        raise FileExistsError("new immutable model-canary root required")
    parent = read(PARENT / "plan.json")
    paths = [NATIVE / "analysis/decision.json", PARENT / "plan.json", PARENT / "inputs.json"]
    paths += [Path(p) for p in parent["bound_files"]]
    refs = {}
    for i in (0, 1):
        p = next((PARENT / f"workers/{i}").glob("[0-9][0-9]-parent.json"))
        row = read(p)
        path = Path(row["logits_path"])
        if row["status"] != "completed" or sha(path) != row["logits_sha256"]:
            raise ValueError("complete immutable storage-only reference required")
        refs[str(i)] = str(path)
        paths += [p, path]
    # Strip historical answer/timing-bearing audit dictionaries. Only frozen
    # controls consumed by request_options enter the new execution plan.
    allowed = set(inspect.signature(request).parameters) | {"uniform_attention_core"}
    record = parent["options"]["mosaickv_exact"]
    options = {k: v for k, v in record.items() if k in allowed}
    options["direct_policy"] = {
        k: record["direct_policy"][k] for k in ("modes", "maximum_added_bytes")
    }
    overlay = dict(parent["component_policies"]["guarded_full"], record_phase_timings=False)
    unlocked = dict(
        overlay,
        inline_repair=False,
        live_anchor_guard=False,
        readout_selection="joint",
        readout_target="fullkv_draft",
        quality_objective="fullkv_draft_improvement",
    )
    policy = {
        "storage_only": asdict(
            OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
        ),
        "storage_codec": asdict(
            OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
        ),
        "hybrid": overlay,
        "hybrid_codec": overlay,
        "draft_hybrid_codec": unlocked,
        "storage_repeat": asdict(
            OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
        ),
    }
    for data in policy.values():
        OverlayPolicy(**dict(data, layers=tuple(data["layers"])))
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_once(root / "inputs.json", read(PARENT / "inputs.json"))
    paths.append(root / "inputs.json")
    with (root / "cpu.txt").open("x") as stream:
        done = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_all_head_source_codec.py",
                "mosaickv/tests/unit/test_all_head_source_gpu.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
                "mosaickv/tests/unit/test_live_anchor_guard.py",
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if done.returncode:
        raise ValueError("CPU mechanism/ownership checks failed")
    paths.append(root / "cpu.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in paths},
            parent=parent["parent"],
            options={"mosaickv_exact": options},
            policies=policy,
            arms=list(policy),
            codec_arms=["storage_codec", "hybrid_codec", "draft_hybrid_codec"],
            references=refs,
            output_predictions=128,
            baseline="storage_only",
            baseline_storage_source_sha="a6f689fcdfae2b9023045aa412c6d90d8ce79012",
            classification="two_consumed_input_factorial_correctness_not_confirmation",
            comparison_type="same_logical_cache_lossless_representation_not_byte_matched_selection",
            seed=20261107,
            labels_accessible_to_planner=False,
            strict_pairwise_checks=[["storage_only", "storage_codec"], ["hybrid", "hybrid_codec"]],
            online_quality_reference_for_unlocked_arm="actual_original_prefill_FullKV_draft",
            unlocked_arm_not_required_to_match_storage_parent_tokens=True,
            independent_quality_or_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    launch(root)


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.all_head_source_gpu import use as source_codec
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import integrated_storage
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("frozen deterministic CUDA launch required")
    torch.set_num_threads(12)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    cfg = RunConfig.from_mapping(plan["parent"]["config"])
    rows, errors, outputs = [], [], {}
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
                config_sha=digest(plan),
                source_dirty=False,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                seed=plan["seed"],
                environment=environment(),
                cuda=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                gpu_count=1,
                backend="huggingface",
                attention="eager",
                precision="bf16",
                measurement_type="validation_smoke_not_paired_latency_evidence",
                reference_logits_sha256=sha(Path(plan["references"][str(i)])),
                **scheduler_identity({}),
            ),
        )
        features = read(root / "inputs.json")[i]
        transport = dict(plan["parent"], options=plan["options"])
        options = campaign.request_options(
            plan["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
        )
        reference = torch.load(plan["references"][str(i)], weights_only=True, map_location="cpu")
        # Warm BOTH codecs before reporting any diagnostic request times.
        for enabled in (False, True):
            with source_codec(enabled):
                warm, _ = campaign.execute(
                    adapter, features, transport, "mosaickv_exact", audit=False
                )
            if not torch.equal(warm.cpu(), reference):
                raise RuntimeError("storage-only warmup does not match frozen original logits")
            del warm
        for arm in plan["arms"]:
            row = dict(
                arm=arm,
                status="failed",
                sample_id=features["sample_id"],
                manifest_path=str(folder / "manifest.json"),
            )
            try:
                gc.collect()
                torch.cuda.synchronize()
                started = perf_counter()
                policy = OverlayPolicy(
                    **dict(plan["policies"][arm], layers=tuple(plan["policies"][arm]["layers"]))
                )
                with (
                    source_codec(arm in plan["codec_arms"]) as codec,
                    integrated_storage(adapter, policy, reader_options=options) as evidence,
                ):
                    values, detail = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=False
                    )
                torch.cuda.synchronize()
                elapsed = perf_counter() - started
                outputs[arm] = values.cpu()
                proof = codec.report()
                if arm in plan["codec_arms"]:
                    if not proof["all_shared_source_consumers_covered"]:
                        raise RuntimeError("replacement did not cover all shared source consumers")
                    if proof["original_owners_retired"] != proof["compressed_replacements"]:
                        raise RuntimeError("replaced original source code owner still retained")
                path = folder / f"{arm}.logits.pt"
                torch.save(outputs[arm], path)
                row.update(
                    status="completed",
                    detail=detail,
                    mechanism_evidence=evidence,
                    source_codec_evidence=proof,
                    request_wall_seconds=elapsed,
                    logits_path=str(path),
                    logits_sha256=sha(path),
                    all_128_parent_logits_bitwise=torch.equal(outputs[arm], reference),
                )
                del values
            except Exception:
                row.update(error=traceback.format_exc())
                errors.append(row["error"])
            rows.append(row)
            write_once(folder / f"{arm}.json", row)
            if errors:
                break
        for a, b in plan["strict_pairwise_checks"]:
            if a not in outputs or b not in outputs or not torch.equal(outputs[a], outputs[b]):
                errors.append(f"lossless factorial parity failed: {a} versus {b}")
        if "storage_repeat" not in outputs or not torch.equal(outputs["storage_repeat"], reference):
            errors.append("model-level A-B-A storage isolation failed")
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            rows=rows,
            errors=errors,
            source_sha=plan["source_sha"],
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    plan = checked(root, expected)
    results = [
        read(root / f"workers/{i}/result.json")
        if (root / f"workers/{i}/result.json").exists()
        else dict(status="missing", errors=["worker missing"], rows=[])
        for i in (0, 1)
    ]
    complete = all(
        r["status"] == "completed" and len(r["rows"]) == len(plan["arms"]) for r in results
    )
    storage_checks = {}
    boundaries = ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")
    if complete:
        for i, result in enumerate(results):
            arms = {row["arm"]: row for row in result["rows"]}
            storage_checks[str(i)] = {
                f"{new}_versus_{old}": {
                    key: arms[new]["detail"][key] < arms[old]["detail"][key] for key in boundaries
                }
                for old, new in (
                    ("storage_only", "storage_codec"),
                    ("hybrid", "hybrid_codec"),
                    ("storage_only", "hybrid_codec"),
                )
            }
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            complete=complete,
            results=results,
            model_parity_gate_passed=complete,
            measured_storage_reductions=storage_checks,
            baseline="evicastkv_storage_only",
            same_128_predictions=True,
            paired_latency_experiment_completed=False,
            independent_quality_or_six_mechanism_superiority_established=False,
            whole_request_KV_peak_certified=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    lines = [
        "# Two consumed-input source/hybrid factorial",
        "",
        "Recorded single-call times, not a paired latency gate. Peak is decode-owned.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Counter peak MiB | Request s | "
        "Parent logits bitwise |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, result in enumerate(results):
        for row in result["rows"]:
            if row["status"] != "completed":
                lines.append(f"| {i} | {row['arm']} | failed | failed | failed | failed | False |")
                continue
            d = row["detail"]
            values = [
                d[k] / 2**20
                for k in (
                    "initial_encoded_bytes",
                    "final_encoded_bytes",
                    "reported_counter_peak_bytes",
                )
            ]
            lines.append(
                f"| {i} | {row['arm']} | {values[0]:.4f} | {values[1]:.4f} | "
                f"{values[2]:.4f} | {row['request_wall_seconds']:.4f} | "
                f"{row['all_128_parent_logits_bitwise']} |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(dict(complete=complete, rows=sum(len(r["rows"]) for r in results))), flush=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--plan-sha")
    a = p.parse_args()
    if a.action == "prepare":
        return prepare(a.root)
    if a.action == "worker":
        return worker(a.root, a.plan_sha)
    if a.action == "analyze":
        return analyze(a.root, a.plan_sha)
    checked(a.root, a.plan_sha)
    return subprocess.call(
        [
            PYTHON,
            str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
            "--policy",
            str(a.root / "admission.json"),
            "--state",
            str(a.root / "admission_state.json"),
            "--apply",
            "--watch",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
