#!/usr/bin/env python3
"""Conditional source-constructor phase localization, no new quality cohort."""

import argparse
import os
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import evicastkv_code_delta_gpu as previous

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = previous.SOURCE
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/lossless-layer-code-delta-gpu-consumed-20261101-retry3"
)


def prepare(root, *, components=False, all_methods=False):
    if root.exists():
        raise FileExistsError("new immutable profile root required")
    plan = previous.previous.read(PARENT / "plan.json")
    jobs = [
        dict(j, reference_worker=j["index"], index=i)
        for i, j in enumerate(
            j for j in plan["jobs"] if previous.common.method_of(j) == "mosaickv_exact"
        )
    ]
    if all_methods:
        jobs = [dict(j, reference_worker=j["index"]) for j in plan["jobs"]]
    bound = dict(plan["bound_files"])
    if components:
        profile_root = PARENT.parent / "source-peak-profile-consumed-20261101-v1"
        proof = previous.previous.read(profile_root / "analysis/decision.json")
        if (
            not proof["complete"]
            or proof["failures"]
            or any(
                case["dominant_peak"]["phase"][-1]
                != "mosaickv.evaluation.shared_source_format.fit_transport"
                for case in proof["cases"]
            )
        ):
            raise ValueError("observed transport-fitting peak must justify this follow-up")
        for p in (profile_root / "plan.json", profile_root / "analysis/decision.json"):
            bound[str(p)] = sha(p)
    if all_methods:
        component_root = PARENT.parent / "componentwise-source-fit-consumed-20261101-v1"
        proof = previous.previous.read(component_root / "analysis/decision.json")
        if (
            not proof["complete"]
            or proof["failures"]
            or any(
                case["dominant_peak"]["phase"][-1]
                != "mosaickv.evaluation.shared_source_format.pack_unprotected"
                for case in proof["cases"]
            )
        ):
            raise ValueError("measured conditional range peak must justify this optimization")
        for p in (component_root / "plan.json", component_root / "analysis/decision.json"):
            bound[str(p)] = sha(p)
        range_root = PARENT.parent / "common-bf16-ranges-consumed-20261101-v1"
        bound[str(range_root / "plan.json")] = sha(range_root / "plan.json")
        for case in (0, 8):
            folder = range_root / f"workers/{case}"
            result = previous.previous.read(folder / "result.json")
            if result["status"] != "completed" or result["failures"]:
                raise ValueError("successful source range profiling required")
            bound[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest_value in result["outputs"].items():
                if sha(folder / name) != digest_value:
                    raise ValueError("profile producer changed")
                bound[str(folder / name)] = digest_value
            observations = previous.previous.read(folder / "observations.json")
            cuda = [v for v in observations["peak_events"] if v["device"].startswith("cuda")]
            if (
                max(cuda, key=lambda v: v["bytes"])["phase"][-1]
                != "mosaickv.evaluation.conditional_precision.pack"
            ):
                raise ValueError("measured quantizer workspace must justify the next optimization")
    for p in (PARENT / "plan.json", SOURCE / "mosaickv/docs/source_peak_profile_20261101.md"):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            plan,
            source_sha=clean_source(SOURCE),
            jobs=jobs,
            bound_files=bound,
            stage="consumed_source_constructor_peak_phase_localization",
            maximum_simultaneous_workers=6 if all_methods else 2,
            maximum_worker_minutes=20 if all_methods else 10,
            profile_only_no_decode_or_new_quality_measurement=True,
            componentwise_transport_fitting=components,
            common_BF16_range_preparation_with_signed_zero_fallback=True,
            all_five_methods_profiled=all_methods,
            compact_quantizer_workspace=True,
        ),
    )
    print(sha(root / "plan.json"))


def submit(root, expected):
    previous.previous.common.inputs(root, expected)
    outcome = previous.previous.read(PARENT / "analysis/decision.json")
    if not outcome["complete"] or outcome["failures"] or not outcome["source_net_storage_gate"]:
        write_once(
            root / "blocked.json",
            dict(
                reason="parent replay/storage proof did not pass",
                parent_decision_sha256=sha(PARENT / "analysis/decision.json"),
                GPU_submitted=False,
            ),
        )
        return
    write_once(
        root / "parent_proof.json",
        dict(
            path=str(PARENT / "analysis/decision.json"),
            sha256=sha(PARENT / "analysis/decision.json"),
        ),
    )
    original = previous.common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_source_peak_profile.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(previous.common.control.parent.subprocess, "check_output", scoped):
        previous.common.control.parent.submit(root, expected)


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation import (
        conditional_precision,
        shared_source_format,
        streaming_baselines,
        streaming_prefill,
        text_source_storage,
    )
    from mosaickv.evaluation.prefill_phase_probe import PhaseCache, PhaseSelectedCache, wrap_phase
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    output = root / f"workers/{index}"
    output.mkdir(parents=True, exist_ok=False)
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
        cfg = RunConfig.from_mapping(previous.previous.resolved_mapping(plan, job))
        manifest = previous.previous.read(
            previous.REFERENCE / f"workers/{job['reference_worker']}/manifest.json"
        )
        manifest.update(
            run_id=str(output),
            parent_run_id=manifest.get("run_id"),
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=digest(dict(plan_sha256=expected, worker=index, variant="phase_profile")),
            evaluator_commit=plan["source_sha"],
            measurement_type="diagnostic_measured",
            environment=previous.common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            independent_quality_N=0,
            whole_request_peak_certified=False,
            configuration_is_unchanged_common_source_precision_variant=True,
            componentwise_transport_fitting=plan.get("componentwise_transport_fitting", False),
        )
        write_once(output / "manifest.json", manifest)
        from mosaickv.evaluation.quantizer_workspace_kernel import gpu_controls

        write_once(output / "quantizer_gpu_controls.json", gpu_controls())
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.previous.read(job["inputs"][0])
        instances = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(PhaseCache, "instances", instances))
            stack.enter_context(patch.object(streaming_prefill, "RetiringPrefillCache", PhaseCache))
            stack.enter_context(
                patch.object(streaming_baselines, "SelectedRetiringCache", PhaseSelectedCache)
            )
            for module, key in (
                (shared_source_format, "construct"),
                (shared_source_format, "fit_transport"),
                (text_source_storage, "optimize"),
                (shared_source_format, "pack_unprotected"),
                (conditional_precision, "pack"),
            ):
                original = getattr(module, key)
                if (
                    module is shared_source_format
                    and key == "fit_transport"
                    and plan.get("componentwise_transport_fitting", False)
                ):
                    original = partial(original, componentwise=True)
                stack.enter_context(
                    patch.object(
                        module,
                        key,
                        wrap_phase(f"{module.__name__}.{key}", original, instances),
                    )
                )
            runtime, packets, memory = previous.construct(
                adapter,
                previous.previous.load_messages(payload),
                tuple(previous.previous.read(Path(plan["profile_path"]))),
                previous.common.method_of(job),
                audit=True,
            )
        reference = previous.previous.read(
            previous.REFERENCE / f"workers/{job['reference_worker']}/offline_payload.json"
        )
        if (
            previous.common.control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
            != reference["payload_sha256"]
        ):
            raise RuntimeError("phase probe changed a frozen payload bit")
        if memory["final_prefill_cache_bytes"] != reference["physical_bytes"]:
            raise RuntimeError("constructor changed physical packet backing storage")
        logits = torch.load(
            previous.REFERENCE / f"workers/{job['reference_worker']}/greedy_logits.pt",
            weights_only=True,
            map_location="cpu",
        )
        if not torch.equal(runtime.logits.cpu().reshape(-1), logits[0].reshape(-1)):
            raise RuntimeError("phase probe changed original prefill logits")
        write_once(
            output / "observations.json",
            dict(
                memory=memory,
                peak_events=instances[0].peak_events,
                payload_bitwise=True,
                physical_packet_bytes_identical=True,
                prefill_logits_bitwise=True,
                no_decode_performed=True,
                unchanged_model_positions_and_quantization=True,
                manifest_path=str(output / "manifest.json"),
            ),
        )
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write_once(output / "result.json", result)


def analyze(root, expected):
    plan = previous.previous.common.inputs(root, expected)
    cases, failures = [], []
    for i in range(len(plan["jobs"])):
        folder = root / f"workers/{i}"
        try:
            result = previous.previous.read(folder / "result.json")
            if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
                raise ValueError("producer provenance mismatch")
            for name, digest in result["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer changed")
            failures.extend(result["failures"])
            if result["status"] != "completed":
                failures.append(dict(case=i, error="producer incomplete"))
            data = previous.previous.read(folder / "observations.json")
            cuda = [r for r in data["peak_events"] if r["device"].startswith("cuda")]
            cases.append(
                dict(case=i, dominant_peak=max(cuda, key=lambda r: r["bytes"]), observations=data)
            )
        except (OSError, ValueError) as exc:
            failures.append(dict(case=i, error=str(exc)))
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(cases) == len(plan["jobs"]) and not failures,
            cases=cases,
            failures=failures,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            superiority_established=False,
            fresh_or_large_campaign_authorized=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "prepare_components",
            "prepare_common_ranges",
            "cpu",
            "submit",
            "worker",
            "analyze",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode in ("prepare", "prepare_components", "prepare_common_ranges"):
        prepare(
            args.root,
            components=args.mode != "prepare",
            all_methods=args.mode == "prepare_common_ranges",
        )
    elif args.mode == "cpu":
        previous.previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
