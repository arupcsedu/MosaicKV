#!/usr/bin/env python3
"""Bounded same-consumed-input counterfactual, veto and isolation verification."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.consumed_guard_check import (
    compare,
    consumed_inputs,
    finite_fault,
    full_head_parity_gate,
    parent_export,
    same_storage,
)
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BENCH = Path("/scratch/djy8hg/runs/evicastkv/device-guard-records-20261107-v1")
PANEL = Path("/scratch/djy8hg/runs/evicastkv/six-component-consumed-panel-20261107-v1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
CANDIDATE = "e3c66f784bef7dfc5f97584e7b847499e24618ac"


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("immutable package boundary required")
    if sha(root / "plan.json") != expected:
        raise ValueError("plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(name)) != value for name, value in p["bound_files"].items()):
        raise ValueError("bound consumed evidence or inputs changed")
    return p


def prepare(root, full_head_parent=False):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable correctness-only output root required")
    if read(BENCH / "analysis/decision.json")["complete"] is not True:
        raise ValueError("prior complete numerical/storage timing study required")
    if read(PANEL / "analysis/panel_stopping_decision.json")["execution_passed"] is not True:
        raise ValueError("previously consumed panel must be complete")
    # The new commit adds only this harness, helpers, tests and documentation.
    # All candidate implementation files must still be byte-identical.
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", CANDIDATE, source], cwd=SOURCE, text=True
    ).splitlines()
    allowed = {
        "mosaickv/scripts/check_consumed_guard.py",
        "mosaickv/slurm/check_consumed_guard.sbatch",
        "mosaickv/src/mosaickv/evaluation/consumed_guard_check.py",
        "mosaickv/tests/unit/test_consumed_guard_check.py",
        "mosaickv/docs/consumed_guard_check.md",
        "mosaickv/scripts/debug_guard_counterfactual.py",
        "mosaickv/slurm/debug_guard_counterfactual.sbatch",
        "mosaickv/tests/unit/test_guard_counterfactual_debug.py",
    }
    if not set(diff) <= allowed:
        raise ValueError("candidate source changed outside correctness harness")
    inputs = consumed_inputs(read(PANEL / "inputs.json"))
    parent = read(BENCH / "plan.json")
    bound = [BENCH / "plan.json", BENCH / "analysis/decision.json"]
    bound += [PANEL / "inputs.json", PANEL / "analysis/panel_stopping_decision.json"]
    bound += [Path(v) for v in parent["bound_files"]]
    bound += [Path(m["path"]) for s in inputs for m in s["media"]]
    policies = {k: dict(v) for k, v in parent["policies"].items()}
    if full_head_parent:
        debug = Path("/scratch/djy8hg/runs/evicastkv/guard-counterfactual-debug-20261107-v1")
        full_head_parity_gate(read(debug / "analysis/decision.json"))
        bound += [debug / "plan.json", debug / "analysis/decision.json"]
        for arm in ("hybrid_reference", "hybrid_fast"):
            policies[arm]["anchor_counterfactual"] = "full_heads"
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as log:
        cpu = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_consumed_guard_check.py",
                "mosaickv/tests/unit/test_device_guard_records.py",
                "mosaickv/tests/unit/test_guard_tail_graph.py",
                "mosaickv/tests/unit/test_guard_readout_reuse.py",
                "mosaickv/tests/unit/test_live_anchor_guard.py",
            ],
            cwd=SOURCE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if cpu.returncode:
        raise RuntimeError("CPU gate failed; no GPU jobs submitted")
    bound.append(root / "cpu.txt")
    write_once(root / "inputs.json", inputs)
    bound.append(root / "inputs.json")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            frozen_candidate_sha=CANDIDATE,
            bound_files={str(p): sha(p) for p in bound},
            parent=parent["parent"],
            options=parent["options"],
            policies=policies,
            inputs=str(root / "inputs.json"),
            expected_samples=16,
            samples_per_worker=8,
            seed=20261107,
            cpu_torch_threads=1,
            classification="consumed_correctness_counterfactual_veto_ABA_not_quality",
            input_annotations_removed=True,
            benchmark_scores_computed=False,
            fresh_or_paper_campaign_authorized=False,
            fault="finite_plus128_to_active_final_hidden_on_fourth_guard_update",
            fault_not_a_quality_or_repair_claim=True,
            timed_latency_claim_authorized=False,
            graph_template_CPU_limit=4 * 2**20,
            full_head_parent=full_head_parent,
            single_head_numerical_optimization_promoted=False,
        ),
    )
    import all_head_native_source as infra

    infra.SOURCE, infra.SHELL = SOURCE, SOURCE / "mosaickv/slurm/check_consumed_guard.sbatch"
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
        maximum_gpu_workers=2,
        cpus_per_worker=12,
        minutes_per_worker=30,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.all_head_source_gpu import use as source_codec
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy, integrated_storage
    from mosaickv.graph.structural_templates import engine_scope
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    p = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("registered deterministic launch required")
    torch.set_num_threads(p["cpu_torch_threads"])
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if index not in (0, 1):
        raise ValueError("only two frozen worker indices")
    folder = root / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    errors, completed = [], 0
    cfg = RunConfig.from_mapping(p["parent"]["config"])
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            frozen_candidate_sha=p["frozen_candidate_sha"],
            config_sha=digest(p),
            source_dirty=False,
            model=asdict(cfg.model),
            dataset=asdict(cfg.dataset),
            environment=environment(),
            seed=p["seed"],
            cuda=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            gpu_count=1,
            backend="huggingface",
            attention="eager",
            precision="bf16",
            measurement_type=p["classification"],
            vllm="not_used",
            sglang="not_used",
            **scheduler_identity({}),
        ),
    )
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
        samples = read(p["inputs"])[index * 8 : (index + 1) * 8]
        first = None
        with engine_scope(p["graph_template_CPU_limit"]) as templates:

            def run(features, arm, fault=False):
                policy = dict(p["policies"][arm])
                # Export only in correctness reference, never in timed production.
                if arm == "hybrid_reference":
                    policy["export_parent_logits"] = True
                policy = OverlayPolicy(**dict(policy, layers=tuple(policy["layers"])))
                gc.collect()
                with ExitStack() as stack:
                    codec = stack.enter_context(
                        source_codec(True, construction="pack_selected", decoder="bitmap_prefix")
                    )
                    exported = (
                        stack.enter_context(parent_export()) if arm == "hybrid_reference" else {}
                    )
                    injections = stack.enter_context(finite_fault(fault))
                    evidence = stack.enter_context(
                        integrated_storage(adapter, policy, reader_options=options)
                    )
                    values, detail = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=False
                    )
                proof = codec.report()
                if (
                    not proof["all_shared_source_consumers_covered"]
                    or proof["compressed_replacements"] != proof["original_owners_retired"]
                ):
                    raise ValueError("all-head source retirement failed")
                if fault and injections != [3]:
                    raise ValueError("one registered fourth-update perturbation required")
                return values.cpu(), detail, evidence, exported.get("logits")

            with (folder / "raw.jsonl").open("x") as stream:
                for features in samples:
                    row = dict(
                        sample_id=features["sample_id"],
                        status="failed",
                        manifest_path=str(folder / "manifest.json"),
                    )
                    try:
                        parent, _, _, _ = run(features, "codec_fast")
                        host, hd, he, counter = run(features, "hybrid_reference")
                        device, dd, de, _ = run(features, "hybrid_fast")
                        controls = compare(parent, host, device, counter, he, de)
                        storage = same_storage(hd, dd)
                        row.update(
                            status="completed",
                            controls=controls,
                            storage=storage,
                            logits_sha256=tensor_sha(device),
                            guard=de["live_anchor_guard"],
                            extra_gpu_bytes=de["extra_gpu_bytes"],
                            residual_cpu_bytes=de["residual_cpu_bytes"],
                        )
                        if first is None:
                            first = dict(features=features, parent=parent, device=device, detail=dd)
                        completed += 1
                    except Exception:
                        row["error"] = traceback.format_exc()
                        errors.append(row["error"])
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps(dict(completed=completed, errors=len(errors))), flush=True)
                    if errors:
                        raise RuntimeError(
                            "stop on failed counterfactual or isolation prerequisite"
                        )
                repeated, rd, _, _ = run(first["features"], "hybrid_fast")
                if not torch.equal(repeated, first["device"]):
                    raise ValueError("A-B-A request isolation changed logits")
                same_storage(first["detail"], rd)
                host, hd, he, counter = run(first["features"], "hybrid_reference", fault=True)
                device, dd, de, _ = run(first["features"], "hybrid_fast", fault=True)
                fault = compare(first["parent"], host, device, counter, he, de)
                same_storage(hd, dd)
                if fault["actual_vetoes"] < 1:
                    raise ValueError("injected finite perturbation did not trigger actual guard")
                write_once(
                    folder / "additional_controls.json",
                    dict(
                        ABA_all_128_logits_bitwise=True,
                        same_engine_between_requests=True,
                        intervening_requests=7,
                        finite_fault=fault,
                        mechanical_not_quality_or_repair=True,
                        template_report=templates.report(),
                    ),
                )
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed", completed_samples=completed, errors=errors
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    p = checked(root, expected)
    results, errors, rows = {}, {}, []
    for i in (0, 1):
        folder = root / f"workers/{i}"
        try:
            result = read(folder / "result.json")
            if result["status"] != "completed" or result["completed_samples"] != 8:
                raise ValueError(result)
            results[str(i)] = read(folder / "additional_controls.json")
            rows += [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        except Exception:
            errors[str(i)] = traceback.format_exc()
    expected_ids = {s["sample_id"] for s in read(p["inputs"])}
    complete = not errors and len(rows) == 16 and {r["sample_id"] for r in rows} == expected_ids
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            complete=complete,
            completed_denominator=len(rows),
            planned_denominator=16,
            results=results,
            errors=errors,
            source_sha=p["source_sha"],
            frozen_candidate_sha=p["frozen_candidate_sha"],
            benchmark_quality_measured=False,
            latency_superiority_measured=False,
            fresh_or_paper_campaign_authorized=False,
            no_new_evaluation_identities=True,
        ),
    )
    print(json.dumps(dict(complete=complete, denominator=len(rows), errors=errors)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--full-head-parent", action="store_true")
    a = parser.parse_args()
    if a.action == "prepare":
        return prepare(a.root, a.full_head_parent)
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
