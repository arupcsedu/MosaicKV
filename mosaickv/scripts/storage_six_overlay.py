#!/usr/bin/env python3
"""Finite same-consumed-input bridge from the validated storage producer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v3")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
ARMS = (
    "parent",
    "disabled_hook",
    "six_overlay",
    "controlled_repair",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
)


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for path, value in plan["bound_files"].items():
        if sha(Path(path)) != value:
            raise ValueError(f"immutable parent/input changed: {path}")
    return plan


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable output root required")
    source_sha = clean_source(SOURCE)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "a6f689f", source_sha], cwd=SOURCE, check=True
    )
    root.mkdir(parents=True)
    paths = [PARENT / name for name in ("plan.json", "options.json", "ledger.json")]
    paths.append(PARENT / "timing/workers/0/raw.jsonl")
    parent = read(PARENT / "plan.json")
    options = read(PARENT / "options.json")["options"]
    inputs = [
        {k: sample[k] for k in ("sample_id", "prompt", "media")}
        for sample in read(PARENT / "ledger.json")["samples"][2:4]
    ]
    write_once(root / "inputs.json", inputs)
    with (root / "cpu.txt").open("x") as stream:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
                "mosaickv/tests/unit/test_streaming_source_constructor.py",
                "mosaickv/tests/unit/test_six_mechanism_gate.py",
                "mosaickv/tests/integration/test_six_mechanism_runtime.py",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=SOURCE,
        )
    write_once(
        root / "cpu.json", dict(passed=result.returncode == 0, log_sha256=sha(root / "cpu.txt"))
    )
    if result.returncode:
        raise RuntimeError("CPU gate failed; no GPU submission")
    paths += [root / "inputs.json", root / "cpu.json", Path(parent["profile_path"])]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source_sha,
            parent=parent,
            options=options,
            bound_files={str(p): sha(p) for p in paths},
            arms=ARMS,
            overlay_policy=dict(
                layers=[31],
                extra_exact_positions=4,
                prototype_members=2,
                maximum_extra_gpu_bytes=16384,
                entropy_threshold=0.5,
            ),
            classification="consumed_storage_extension_diagnostic_not_confirmation",
            prototype_policy="shared_phase_safe_member_overlay_not_original_slot_compaction",
            output_tokens=128,
            guard_tokens=32,
            guard_kl=0.02,
            guard_max_logit=1.0,
            larger_campaign_authorized=False,
            parent_reader_options_unchanged=True,
        ),
    )
    import six_mechanism_feasibility as infra

    infra.SHELL = SOURCE / "mosaickv/slurm/storage_six_overlay.sbatch"
    infra.launch(root, "canary")


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy, integrated_storage
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic CUDA launch policy missing")
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
                visible_gpu_count=1,
                backend="huggingface",
                attention="eager",
                precision="bf16",
                seed=plan["parent"]["protocol"]["seed"],
                measurement_type="validation_smoke",
                **scheduler_identity({}),
                parent_ledger_sha=sha(PARENT / "ledger.json"),
            ),
        )
        inputs = read(root / "inputs.json")[index]
        transport_plan = dict(plan["parent"], options=plan["options"])
        options = campaign.request_options(
            plan["options"]["mosaickv_exact"], transport_plan, "mosaickv_exact", False
        )
        values_by_arm = {}
        historical = [
            json.loads(line)
            for line in (PARENT / "timing/workers/0/raw.jsonl").read_text().splitlines()
        ]
        for arm in ARMS:
            gc.collect()
            row = dict(
                arm=arm,
                sample_id=inputs["sample_id"],
                status="failed",
                manifest_path=str(folder / "manifest.json"),
            )
            try:
                if arm in ("disabled_hook", "six_overlay", "controlled_repair"):
                    policy = OverlayPolicy(
                        **dict(
                            plan["overlay_policy"],
                            layers=tuple(plan["overlay_policy"]["layers"]),
                            enabled=arm != "disabled_hook",
                            force_repair=arm == "controlled_repair",
                        )
                    )
                    with integrated_storage(adapter, policy, reader_options=options) as proof:
                        values, detail = campaign.execute(
                            adapter, inputs, transport_plan, "mosaickv_exact", audit=False
                        )
                    row["mechanism_evidence"] = proof
                    detail["original_six_mechanisms_disabled"] = arm == "disabled_hook"
                    detail["effective_method"] = "evicastkv_storage_six_precision_overlay_v1"
                else:
                    method = "mosaickv_exact" if arm == "parent" else arm
                    values, detail = campaign.execute(
                        adapter, inputs, transport_plan, method, audit=False
                    )
                if arm == "disabled_hook":
                    if not torch.equal(values, values_by_arm["parent"]):
                        raise RuntimeError("disabled bridge changed a parent logit bit")
                    row["disabled_hook_bitwise_parent"] = True
                if arm == "parent":
                    reference = next(
                        r
                        for r in historical
                        if r["method"] == "mosaickv_exact" and r["sample_id"] == inputs["sample_id"]
                    )
                    if detail["generated_token_ids"] != reference["generated_token_ids"]:
                        raise RuntimeError("winning historical parent tokens changed")
                    row["historical_winning_parent_tokens_identical"] = True
                if arm in ("six_overlay", "controlled_repair"):
                    if not proof["layers"] or not proof["all_residuals_pinned"]:
                        raise RuntimeError("all mechanisms and pinned residuals must be real")
                    if arm == "controlled_repair" and not proof["repair_events"]:
                        raise RuntimeError("actual repair required for stress control")
                    original = values_by_arm["parent"][:32].double()
                    actual = values[:32].double()
                    kl = (
                        original.softmax(-1) * (original.log_softmax(-1) - actual.log_softmax(-1))
                    ).sum(-1)
                    row["parent_guard"] = dict(
                        maximum_kl=float(kl.max()),
                        maximum_logit_difference=float((original - actual).abs().max()),
                        token_agreement=float(
                            (original.argmax(-1) == actual.argmax(-1)).double().mean()
                        ),
                        exact_shared_prefix_comparison=False,
                        same_parent_prefill_controls=True,
                    )
                values_by_arm[arm] = values
                row.update(status="completed", detail=detail)
            except Exception:
                row["error"] = traceback.format_exc()
                errors.append(row["error"])
            write_once(folder / f"{len(rows):02}-{arm}.json", row)
            rows.append(row)
            if row["status"] != "completed":
                break
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
    plan = checked(root, expected)
    rows, errors = [], []
    for index in (0, 1):
        folder = root / f"workers/{index}"
        result = read(folder / "result.json") if (folder / "result.json").exists() else {}
        if result.get("status") != "completed":
            errors += result.get("errors", [f"worker {index} missing or failed"])
        rows += [read(p) for p in sorted(folder.glob("[0-9][0-9]-*.json"))]
    if len(rows) != 2 * len(ARMS):
        errors.append(f"incomplete {len(rows)}/{2 * len(ARMS)}")
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(out / "rows.json", rows)
    stable = (
        all(
            r.get("parent_guard", {}).get("maximum_kl", float("inf")) <= plan["guard_kl"]
            and r.get("parent_guard", {}).get("maximum_logit_difference", float("inf"))
            <= plan["guard_max_logit"]
            and r.get("parent_guard", {}).get("token_agreement", 0) == 1
            for r in rows
            if r["arm"] == "six_overlay"
        )
        and len([r for r in rows if r["arm"] == "six_overlay" and r["status"] == "completed"]) == 2
    )
    write_once(
        out / "decision.json",
        dict(
            execution_passed=not errors,
            preliminary_parent_stability_passed=stable,
            errors=errors,
            row_count=len(rows),
            source_sha=plan["source_sha"],
            larger_campaign_authorized=False,
            superiority_established=False,
            classification=plan["classification"],
        ),
    )
    lines = [
        "# Storage-parent six-mechanism extension",
        "",
        f"Execution complete: {not errors}; {len(rows)}/{2 * len(ARMS)} rows.",
        "",
        "Consumed inputs; different encoded operating points, not byte-matched evidence.",
        "Shared prototype overlays preserve all parent positions; not slot compaction.",
        "",
        "| Worker | Arm | Initial MiB | Final MiB | Decode-owned peak MiB | Request s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "completed":
            continue
        d = row["detail"]
        lines.append(
            f"| {row['manifest_path'].split('/')[-2]} | {row['arm']} | "
            f"{d['initial_encoded_bytes'] / 2**20:.3f} | {d['final_encoded_bytes'] / 2**20:.3f} | "
            f"{d['reported_counter_peak_bytes'] / 2**20:.3f} | {d['end_to_end_seconds']:.3f} |"
        )
    for error in errors:
        lines += ["", "```", error, "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if not errors:
        from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
        from mosaickv.evaluation.metrics.types import MetricSample

        refs = {
            s["sample_id"]: tuple(s["references"]) for s in read(PARENT / "ledger.json")["samples"]
        }
        groups = {
            arm: [
                MetricSample(r["sample_id"], r["detail"]["answer"], refs[r["sample_id"]])
                for r in rows
                if r["arm"] == arm
            ]
            for arm in ARMS
        }
        scorer = TokenizedCider([s for part in groups.values() for s in part])
        write_once(
            out / "diagnostic_cider.json",
            dict(
                scores={arm: scorer(samples) for arm, samples in groups.items()},
                sample_count=2,
                consumed_development_only=True,
                superiority_established=False,
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--phase", default="canary")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root)
    elif a.action == "worker":
        return worker(a.root, a.plan_sha)
    elif a.action == "analyze":
        analyze(a.root, a.plan_sha)
    else:
        checked(a.root, a.plan_sha)
        return subprocess.call(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(a.root / "canary_admission.json"),
                "--state",
                str(a.root / "canary_admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
