#!/usr/bin/env python3
"""Consumed numerical failure localization, never a quality or timing sweep."""

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.consumed_guard_check import consumed_inputs, parent_export
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BENCH = Path("/scratch/djy8hg/runs/evicastkv/device-guard-records-20261107-v1")
FAILED = Path("/scratch/djy8hg/runs/evicastkv/device-guard-consumed-check-20261107-v1")
PANEL = Path("/scratch/djy8hg/runs/evicastkv/six-component-consumed-panel-20261107-v1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def delta(reference, other):
    import torch

    if reference.shape != other.shape or reference.ndim != 2:
        raise ValueError("aligned complete vocabulary matrices required")
    if not torch.isfinite(reference).all() or not torch.isfinite(other).all():
        raise ValueError("nonfinite logits")
    changed = (reference != other).any(-1).nonzero().flatten().tolist()
    tokens = (reference.argmax(-1) != other.argmax(-1)).nonzero().flatten().tolist()
    return dict(
        all_vectors_bitwise=not changed,
        changed_vector_count=len(changed),
        changed_prefix_indices=changed,
        maximum_absolute_difference=float((reference - other).abs().max()),
        changed_greedy_prefix_indices=tokens,
        greedy_agreement=1 - len(tokens) / len(reference),
    )


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("immutable package boundary required")
    if sha(root / "plan.json") != expected:
        raise ValueError("registered plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(k)) != v for k, v in p["bound_files"].items()):
        raise ValueError("immutable consumed controls changed")
    return p


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable failure diagnostic required")
    source = clean_source(SOURCE)
    p = read(BENCH / "plan.json")
    inputs = consumed_inputs(read(PANEL / "inputs.json"))
    # These are numerical-failure identities, not answer/score-selected inputs.
    chosen = [inputs[0], inputs[9]]
    failed_rows = []
    for i in (0, 1):
        failed_rows += [
            json.loads(x) for x in (FAILED / f"workers/{i}/raw.jsonl").read_text().splitlines()
        ]
    failed_ids = {r["sample_id"] for r in failed_rows if r["status"] == "failed"}
    if {s["sample_id"] for s in chosen} != failed_ids:
        raise ValueError("only the two recorded consumed numerical failures are permitted")
    paths = [BENCH / "plan.json", PANEL / "inputs.json", FAILED / "analysis/decision.json"]
    paths += [Path(v) for v in p["bound_files"]]
    paths += [FAILED / f"workers/{i}/raw.jsonl" for i in (0, 1)]
    paths += [Path(m["path"]) for s in chosen for m in s["media"]]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_once(root / "inputs.json", chosen)
    paths.append(root / "inputs.json")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(k): sha(k) for k in paths},
            parent=p["parent"],
            options=p["options"],
            policies=p["policies"],
            inputs=str(root / "inputs.json"),
            seed=20261107,
            measurement_type="same_consumed_inputs_single_vs_full_head_counterfactual_diagnostic",
            benchmark_labels_accessed=False,
            thresholds_changed=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    import all_head_native_source as infra

    infra.SOURCE, infra.SHELL = SOURCE, SOURCE / "mosaickv/slurm/debug_guard_counterfactual.sbatch"
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
    write_once(
        root / "submission.json",
        dict(
            array=array,
            controller=infra.submit(root, "controller"),
            analyzer=infra.submit(root, "analyze", after=array),
        ),
    )
    print(json.dumps(read(root / "submission.json")), flush=True)


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
        raise ValueError("registered deterministic CUDA launch required")
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    cfg = RunConfig.from_mapping(p["parent"]["config"])
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
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
            measurement_type=p["measurement_type"],
            vllm="not_used",
            sglang="not_used",
            **scheduler_identity({}),
        ),
    )
    errors = []
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
        features = read(p["inputs"])[i]
        tensors, rows = {}, {}
        with engine_scope():
            for name in ("parent", "single", "full"):
                arm = "codec_fast" if name == "parent" else "hybrid_reference"
                policy = dict(p["policies"][arm])
                if name != "parent":
                    policy.update(
                        export_parent_logits=True,
                        anchor_counterfactual="full_heads" if name == "full" else "single_head",
                    )
                policy = OverlayPolicy(**dict(policy, layers=tuple(policy["layers"])))
                gc.collect()
                with (
                    source_codec(True, construction="pack_selected", decoder="bitmap_prefix"),
                    parent_export() as capture,
                    integrated_storage(adapter, policy, reader_options=options) as evidence,
                ):
                    values, detail = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=False
                    )
                tensors[name] = values.cpu()
                if name != "parent":
                    tensors[name + "_counterfactual"] = capture["logits"]
                rows[name] = dict(detail=detail, mechanism_evidence=evidence)
        comparisons = {}
        for name in ("single", "full"):
            comparisons[name + "_counterfactual_vs_parent"] = delta(
                tensors["parent"][1:], tensors[name + "_counterfactual"]
            )
            comparisons[name + "_output_vs_parent"] = delta(tensors["parent"], tensors[name])
        comparisons["single_vs_full_counterfactual"] = delta(
            tensors["full_counterfactual"], tensors["single_counterfactual"]
        )
        path = folder / "logits.pt"
        torch.save(tensors, path)
        write_once(
            folder / "numerical.json",
            dict(
                sample_id=features["sample_id"],
                comparisons=comparisons,
                raw_arms=rows,
                logits_sha256=sha(path),
                no_quality_or_latency_claim=True,
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json", dict(status="failed" if errors else "completed", errors=errors)
    )
    return int(bool(errors))


def analyze(root, expected):
    p = checked(root, expected)
    results, errors = {}, {}
    for i in (0, 1):
        try:
            r = read(root / f"workers/{i}/result.json")
            if r["status"] != "completed":
                raise ValueError(r)
            numerical = read(root / f"workers/{i}/numerical.json")
            results[str(i)] = numerical["comparisons"]
        except Exception:
            errors[str(i)] = traceback.format_exc()
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            complete=not errors and len(results) == 2,
            results=results,
            errors=errors,
            source_sha=p["source_sha"],
            all_tolerances_unchanged=True,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    print(json.dumps(dict(complete=not errors, comparisons=results, errors=errors)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    a = parser.parse_args()
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
