#!/usr/bin/env python3
"""One-A100, two-consumed-input retained-path phase profile; not a latency trial."""

import argparse
import cProfile
import json
import os
import pstats
import subprocess
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BENCH = Path("/scratch/djy8hg/runs/evicastkv/direct-structural-key-GPU-20261107-v3-feasible-groups")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
CANDIDATE = "1bfbd01e9cb1454839efbfc78722afe7feaaaee2"


def read(p):
    return json.loads(Path(p).read_text())


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("immutable package boundary required")
    if sha(root / "plan.json") != expected:
        raise ValueError("phase-profile plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(k)) != v for k, v in p["bound_files"].items()):
        raise ValueError("bound consumed inputs or model references changed")
    return p


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable profile-only root required")
    if read(BENCH / "analysis/decision.json")["complete"] is not True:
        raise ValueError("complete retained-path benchmark controls required")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", CANDIDATE, source], cwd=SOURCE, text=True
    ).splitlines()
    allowed = {
        "mosaickv/scripts/profile_retained_six_stage.py",
        "mosaickv/slurm/profile_retained_six_stage.sbatch",
        "mosaickv/scripts/audit_six_stage_impact.py",
        "mosaickv/src/mosaickv/evaluation/six_stage_impact.py",
        "mosaickv/tests/unit/test_six_stage_impact.py",
    }
    if not set(changed) <= allowed:
        raise ValueError("retained candidate implementation must remain byte-identical")
    p = read(BENCH / "plan.json")
    policy = dict(p["policies"]["hybrid_reference"])
    if (
        policy.get("graph_direct_metadata_key", False)
        or policy["anchor_counterfactual"] != "full_heads"
    ):
        raise ValueError("retained reference policy, not the negative experimental arm, required")
    paths = [BENCH / "plan.json", BENCH / "analysis/decision.json"]
    paths += [Path(k) for k in p["bound_files"]]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            candidate_implementation_sha=CANDIDATE,
            bound_files={str(k): sha(k) for k in paths},
            parent=p["parent"],
            options=p["options"],
            inputs=p["inputs"],
            references=p["references"],
            policy=policy,
            output_predictions=128,
            samples=2,
            CPU_threads=1,
            seed=p["seed"],
            warmups_per_input=1,
            classification="retained_path_consumed_synchronized_phase_profile_not_latency_trial",
            repair_timing_scope="first_update_policy_check_and_optional_recompute_excluding_key_snapshot",
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    import all_head_native_source as infra

    infra.SOURCE, infra.SHELL = SOURCE, SOURCE / "mosaickv/slurm/profile_retained_six_stage.sbatch"
    worker = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--account=bii_dsc_community",
            "--nodes=1",
            "--ntasks=1",
            "--job-name=mk-retained-stage-profile",
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:15:00",
            "--array=0%1",
            "--hold",
            f"--output={root}/logs/worker-%A_%a.out",
            f"--error={root}/logs/worker-%A_%a.err",
            str(infra.SHELL),
            str(SOURCE),
            str(root),
            sha(root / "plan.json"),
            "worker",
        ],
        text=True,
    ).strip()
    write_once(
        root / "admission.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[worker],
            adopt_existing_holds=[worker],
        ),
    )
    receipt = dict(
        array=worker,
        controller=infra.submit(root, "controller"),
        analyzer=infra.submit(root, "analyze", after=worker),
        maximum_gpu_workers=1,
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
    from mosaickv.evaluation.inline_first_step_repair import InlineRepair
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
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    cfg = RunConfig.from_mapping(p["parent"]["config"])
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
        base = OverlayPolicy(**dict(p["policy"], layers=tuple(p["policy"]["layers"])))
        with engine_scope():
            for i, features in enumerate(read(p["inputs"])):
                folder = root / f"workers/{i}"
                folder.mkdir(parents=True, exist_ok=False)
                write_once(
                    folder / "manifest.json",
                    dict(
                        git_sha=p["source_sha"],
                        candidate_implementation_sha=CANDIDATE,
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
                        sample_id=features["sample_id"],
                        vllm="not_used",
                        sglang="not_used",
                        **scheduler_identity({}),
                    ),
                )
                wanted = torch.load(
                    p["references"][str(i)]["hybrid"], map_location="cpu", weights_only=True
                )
                with (
                    source_codec(True, construction="pack_selected", decoder="bitmap_prefix"),
                    integrated_storage(
                        adapter,
                        replace(base, record_phase_timings=False, verify_anchor_suffix=False),
                        reader_options=options,
                    ),
                ):
                    warm, _ = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=False
                    )
                if not torch.equal(warm.cpu(), wanted):
                    raise ValueError("warmup changed a frozen vocabulary vector")
                del warm
                repairs = []
                original = InlineRepair.finish

                def finish(owner, *args, _original=original, _measurements=repairs):
                    torch.cuda.synchronize()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    start = perf_counter()
                    result = _original(owner, *args)
                    end.record()
                    torch.cuda.synchronize()
                    _measurements.append(
                        dict(
                            wall_seconds=perf_counter() - start,
                            GPU_stream_interval_ms=begin.elapsed_time(end),
                            includes_host_gaps_not_pure_kernel_time=True,
                            repair_events=len(owner.report["repair_events"]),
                        )
                    )
                    return result

                profiler = cProfile.Profile()
                with (
                    patch.object(InlineRepair, "finish", finish),
                    source_codec(True, construction="pack_selected", decoder="bitmap_prefix"),
                    integrated_storage(
                        adapter,
                        replace(base, record_phase_timings=True, verify_anchor_suffix=True),
                        reader_options=options,
                    ) as evidence,
                ):
                    values, detail = profiler.runcall(
                        campaign.execute,
                        adapter,
                        features,
                        transport,
                        "mosaickv_exact",
                        audit=False,
                    )
                if not torch.equal(values.cpu(), wanted) or len(repairs) != 1:
                    raise ValueError(
                        "profile changed logits or omitted the actual first-update policy"
                    )
                if (
                    evidence["live_anchor_guard"]["early_middle_late_native_suffix_verifications"]
                    != 3
                ):
                    raise ValueError("native suffix parity controls missing")
                profiler.dump_stats(str(folder / "profile.pstats"))
                with (folder / "profile.txt").open("x") as out:
                    pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(100)
                write_once(
                    folder / "profile.json",
                    dict(
                        all_128_logits_bitwise=True,
                        labels_accessed=False,
                        detail=detail,
                        mechanism_evidence=evidence,
                        profiled_source_sha=p["source_sha"],
                        candidate_implementation_sha=CANDIDATE,
                        policy_role="retained_hybrid_reference_not_negative_metadata_arm",
                        repair_policy_phase_seconds=repairs[0]["wall_seconds"],
                        repair_phase=repairs[0],
                        timing_not_an_uninstrumented_request_trial=True,
                    ),
                )
                del values
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        root / "result.json", dict(status="failed" if errors else "completed", errors=errors)
    )
    return int(bool(errors))


def analyze(root, expected):
    p = checked(root, expected)
    errors = []
    try:
        result = read(root / "result.json")
        if result["status"] != "completed":
            raise ValueError(result)
        for i in (0, 1):
            if read(root / f"workers/{i}/profile.json")["all_128_logits_bitwise"] is not True:
                raise ValueError("missing complete profile")
    except Exception:
        errors.append(traceback.format_exc())
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            complete=not errors,
            errors=errors,
            source_sha=p["source_sha"],
            candidate_implementation_sha=CANDIDATE,
            profile_only=True,
            quality_or_latency_superiority_established=False,
        ),
    )
    print(json.dumps(dict(complete=not errors, errors=errors)), flush=True)


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
