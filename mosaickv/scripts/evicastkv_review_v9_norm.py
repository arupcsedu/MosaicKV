#!/usr/bin/env python3
"""Finite V9 correction and saved-evidence audits; no new cohort or quality run."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
V8 = Path("/scratch/djy8hg/audits/fox-review-v8-consumed-diagnostics-20261110")
V6 = Path("/scratch/djy8hg/runs/evicastkv/fox-review-v6-parallel-controls-20261110")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/fox-review-v4-native-coco-20261109")
BASE_SHA = "c0fc23bc37e67a7650f7735da3e84e57342350d1"
MODULES = dict(
    norm="evicastkv_boundary_shape_validation",
    gaps="evicastkv_freshness_gap_audit",
    line_recovery="evicastkv_freshness_line_recovery",
    status="evicastkv_review_status_refresh",
)


def module(name):
    return importlib.import_module(MODULES[name])


def submit(root, label, mode, dependencies=(), *, gpu=False, tests=False):
    return shared.submit(
        root,
        label,
        mode,
        dependencies,
        gpu=gpu,
        tests=tests,
        launcher="evicastkv_review_v9_norm",
        job_prefix="fox9",
    )


def register(root, *, execute=False):
    root = Path(root).resolve()
    audits = Path("/scratch/djy8hg/audits").resolve()
    if root == audits or not root.is_relative_to(audits) or root.exists():
        raise ValueError("a new write-once scratch audit root is required")
    revision = clean_source(SOURCE)
    parent = shared.read(V8 / "registration.json")
    clean_source(Path(parent["source_root"]), BASE_SHA)
    if parent["source_sha"] != BASE_SHA:
        raise ValueError("wrong immutable V8 source")
    final = shared.read(V8 / "FINAL_report.json")
    if final["errors"] or final["missing_artifacts"]:
        raise ValueError("V8 must complete without missing diagnostic artifacts")
    paths = [
        V8 / "registration.json",
        V8 / "FINAL_report.json",
        V8 / "rmsnorm/GATE_native_rmsnorm_replay.json",
        V8 / "freshness-continuation/aggregate/inventory.json",
        V8 / "protection/analysis/GATE_source_protection.json",
        SOURCE / "mosaickv/scripts/evicastkv_review_v9_norm.py",
        SOURCE / "mosaickv/docs/fox_review_v9_shape_normalization.md",
    ]
    paths += [SOURCE / f"mosaickv/scripts/{value}.py" for value in MODULES.values()]
    write_once(
        root / "registration.json",
        dict(
            source_root=str(SOURCE),
            source_sha=revision,
            execute=execute,
            bindings={str(path): sha(path) for path in paths},
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            workers=1,
            no_new_inputs=True,
            no_labels=True,
            no_output_patch=True,
            native_RMSNorm_forward_weights_epsilon_and_dtype_preserved=True,
            changed_boundary_launch_geometry_explicitly_labeled=True,
            ordinary_reference_and_postboundary_decoding=True,
            production_defaults_changed=False,
            original_logit_atol=1e-4,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
            freeze_authorized=False,
            superiority_claim_authorized=False,
        ),
    )
    if not execute:
        return dict(status="dry_run", submitted_jobs=0)
    try:
        cpu = submit(root, "cpu", "tests", tests=True)
        return dict(cpu=cpu, prepare=submit(root, "prepare", "prepare", [cpu]))
    except shared.SubmissionError as error:
        return shared.submission_block(root, "v9-registration", error)


def load(root):
    lock = shared.read(root / "registration.json")
    clean_source(SOURCE, lock["source_sha"])
    if lock["execute"] is not True:
        raise ValueError("dry registration cannot execute")
    for name, expected in lock["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"bound review artifact changed: {name}")
    return lock


def prepare(root):
    lock = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, lock["source_sha"]))
    except Exception as error:
        shared.failure(root, "cpu", error)
        return finish(root)
    try:
        module("gaps").run(root / "freshness-gaps", V8 / "freshness-continuation")
    except Exception as error:
        shared.failure(root, "gaps", error)
    try:
        module("line_recovery").run(
            root / "freshness-line-recovery",
            V8 / "freshness-continuation",
            max_records=4,
            cpu_gate_root=root,
            inventory_sha256=lock["bindings"][
                str(V8 / "freshness-continuation/aggregate/inventory.json")
            ],
        )
    except Exception as error:
        shared.failure(root, "line-recovery", error)
    try:
        preparation = module("norm").prepare(
            root / "norm",
            PARENT,
            V6,
            native_norm_boundary=True,
            norm_evidence=V8 / "rmsnorm/GATE_native_rmsnorm_replay.json",
        )
    except Exception as error:
        shared.failure(root, "norm-prepare", error)
        return finish(root)
    try:
        snapshot = shared.gpu_snapshot()
        dependencies = sorted(
            {r["job"].split("_")[0] for r in snapshot["active"] + snapshot["pending"]}
        )
        write_once(
            root / "admission.json",
            dict(
                snapshot=snapshot,
                preparation=preparation,
                wait_for_existing_own_gpu_jobs=dependencies,
                workers=1,
                maximum_a100s=6,
            ),
        )
    except Exception as error:
        shared.failure(root, "gpu-admission", error)
        return finish(root)
    worker = submit(root, "norm", "norm-worker", dependencies, gpu=True)
    return dict(worker=worker, analysis=submit(root, "analysis", "analyze", [worker]))


def finish(root):
    lock = load(root)
    diagnostics = dict(
        numerical=root / "norm/GATE_shape_transport.json",
        freshness_gaps=root / "freshness-gaps/GATE_freshness_gaps.json",
        freshness_line_recovery=root / "freshness-line-recovery/GATE_freshness_line_recovery.json",
    )
    try:
        module("status").run(root / "review-status", current_diagnostics=diagnostics)
    except Exception as error:
        shared.failure(root, "review-status", error)
    files = [*diagnostics.values(), root / "review-status/GATE_review_status.json"]
    errors = {str(p): shared.read(p) for p in sorted((root / "errors").glob("*.json"))}
    write_once(
        root / "FINAL_report.json",
        dict(
            source_sha=lock["source_sha"],
            status="completed_with_blocked_controls"
            if errors or not all(p.exists() for p in files)
            else "completed_bounded_review_correction_not_confirmation",
            artifacts={
                str(p): dict(sha256=sha(p), value=shared.read(p))
                if p.exists()
                else dict(status="missing_or_blocked")
                for p in files
            },
            errors=errors,
            original_negative_results_preserved=True,
            superiority_established=False,
            configuration_freeze_authorized=False,
            fresh_confirmation_authorized=False,
            no_automatic_quality_or_large_campaign=True,
        ),
    )
    return shared.read(root / "FINAL_report.json")


def run(root, mode):
    load(root)
    try:
        if mode == "prepare":
            return prepare(root)
        if mode == "norm-worker":
            if shared.gpu_snapshot()["active_gpus"] > 6:
                raise ValueError("six-A100 aggregate limit exceeded")
            return module("norm").worker(root / "norm")
        if mode == "analyze":
            try:
                module("norm").analyze(root / "norm")
            except Exception as error:
                shared.failure(root, "norm-analysis", error)
            return finish(root)
        raise ValueError("unregistered V9 stage")
    except shared.SubmissionError as error:
        return shared.submission_block(root, mode, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "prepare", "norm-worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else run(args.root, args.mode)
    )
    print(json.dumps(value, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
