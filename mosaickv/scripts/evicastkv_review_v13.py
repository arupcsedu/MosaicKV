#!/usr/bin/env python3
"""Finite consumed-input repairs; no automatic fresh cohort or paper claim."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1")
PARENT_SHA = "76d4ced1a10c6746bad73f564e84210522355ef9"
LANES = {
    "lookm": ("evicastkv_lookm_hf_checkpoint", "GATE_lookm_hf_checkpoint_compat.json"),
    "streaming": ("evicastkv_streaming_lossless_source", "GATE_streaming_lossless_source.json"),
    "ownership": ("evicastkv_source_ownership_v12", "GATE_source_ownership_v12.json"),
    "providers": (
        "evicastkv_historical_provider_superset",
        "GATE_historical_provider_superset.json",
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def module(lane):
    return importlib.import_module(LANES[lane][0])


def load(root):
    plan = read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["execute"] is not True or plan["confirmation_authorized"] is not False:
        raise ValueError("Only the registered finite diagnostic is executable")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Immutable evidence changed: {path}")
    return plan


def submit(root, label, mode, *, dependencies=(), gpu=False, array=False, tests=False):
    if gpu and (array or tests):
        raise ValueError("Each independent diagnostic uses one GPU")
    if (root / f"jobs/{label}-request.json").exists():
        raise FileExistsError("No blind resubmission of an attempted scheduler request")
    numerical = mode in {"after-streaming", "after-ownership", "after-lookm"}
    cpus = 12 if gpu or tests or numerical else 1 if array else 4
    minutes = 20 if gpu or tests else 5 if array else 15
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox13-{label}",
        f"--cpus-per-task={cpus}",
        "--mem=64G" if gpu else "--mem=16G",
        f"--time=00:{minutes:02d}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
        "--partition=bii-gpu" if gpu else "--partition=bii",
    ]
    if gpu:
        command += ["--reservation=bi_fox_dgx", "--gres=gpu:a100:1"]
    if array:
        command += ["--array=0-7%8"]
    if dependencies:
        command += ["--dependency=afterany:" + ":".join(shared.identity(d) for d in dependencies)]
    launcher = "evicastkv_review_closure_v2" if tests else "evicastkv_review_v13"
    command += [
        str(SOURCE / f"mosaickv/slurm/{launcher}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/{label}-request.json", dict(command=command, single_attempt=True))
    try:
        job = shared.identity(subprocess.check_output(command, text=True, stderr=subprocess.PIPE))
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                stderr=str(getattr(error, "stderr", "")),
                automatic_retry=False,
            ),
        )
        raise shared.SubmissionError("Failed or unacknowledged V13 submission") from error
    write_once(root / f"jobs/{label}-submission.json", dict(job=job, command=command))
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    scratch = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == scratch or not root.is_relative_to(scratch):
        raise ValueError("New write-once scratch workspace output root required")
    revision = clean_source(SOURCE)
    parent = read(PARENT / "registration.json")
    if parent["source_sha"] != PARENT_SHA:
        raise ValueError("Wrong immutable parent campaign")
    clean_source(Path(parent["source_root"]), PARENT_SHA)
    gate = read(PARENT / "lossless/GATE_lossless_source_control.json")
    required = (
        "all_original_prompt_KV_bitwise",
        "all_128_source_tokens_agree",
        "all_unchanged_32_token_guards_pass",
        "both_initial_caches_smaller_than_raw",
    )
    if (
        gate["source_sha"] != PARENT_SHA
        or gate["failures"]
        or any(gate.get(key) is not True for key in required)
    ):
        raise ValueError("Streaming follow-up requires the actual passed lossless source control")
    paths = [
        PARENT / "registration.json",
        PARENT / "lossless/GATE_lossless_source_control.json",
        PARENT / "lookm/GATE_lookm_hf_checkpoint_compat.json",
        PARENT / "lookm/worker/result.json",
        PARENT / "lookm/worker/gpu_environment.json",
        PARENT / "ownership/worker/result.json",
        PARENT / "ownership/protocol.json",
        SOURCE / "mosaickv/env/common/requirements.lock",
        Path(__file__),
    ]
    paths += [SOURCE / f"mosaickv/scripts/{name}.py" for name, _ in LANES.values()]
    paths += [
        SOURCE / "mosaickv/scripts/evicastkv_history_v13.py",
        SOURCE / "mosaickv/slurm/evicastkv_history_v13.sbatch",
    ]
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            bindings={str(p): sha(p) for p in paths},
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            maximum_new_gpu_workers=3,
            consumed_inputs_only=True,
            production_three_bit_codec_unchanged=True,
            benchmark_labels_forbidden=True,
            registered_model_guards_unchanged=True,
            confirmation_authorized=False,
            paper_campaign_authorized=False,
            fresh_cohort_selection_authorized=False,
            automatic_retry=False,
            maximum_additional_history_segments=24,
            maximum_parent_history_hops=4,
        ),
    )
    if not execute:
        return dict(status="dry_run", jobs=0)
    cpu = submit(root, "cpu", "tests", tests=True)
    return dict(cpu=cpu, prepare=submit(root, "prepare", "prepare", dependencies=[cpu]))


def prepare(root):
    plan = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, plan["source_sha"]))
    except Exception as error:
        shared.failure(root, "cpu", error)
        return finish(root)
    # Capture admission once before submitting any of our three independent
    # GPU diagnostics; otherwise later workers accidentally serialize behind
    # earlier workers from this same controller.
    snapshot = shared.gpu_snapshot()
    waits = sorted({r["job"].split("_")[0] for r in snapshot["active"] + snapshot["pending"]})
    write_once(
        root / "GPU-admission.json",
        dict(
            snapshot=snapshot,
            wait_for_existing_own_gpu_jobs=waits,
            maximum_new_gpu_workers=3,
            maximum_aggregate_a100s=6,
            no_other_users_jobs_modified=True,
        ),
    )
    results, analyses = {}, []
    for lane in LANES:
        try:
            kwargs = dict(cpu_gate_root=root) if lane != "ownership" else {}
            results[lane] = module(lane).prepare(root / lane, **kwargs)
            gpu = lane != "providers"
            job = submit(
                root,
                lane,
                f"{lane}-worker",
                dependencies=waits if gpu else (),
                gpu=gpu,
                array=not gpu,
            )
            callback = submit(root, f"after-{lane}", f"after-{lane}", dependencies=[job])
            analyses.append(callback)
        except Exception as error:
            shared.failure(root, f"prepare-{lane}", error)
    write_once(root / "preparation.json", results)
    # If even one callback submission fails, its producer is also a dependency
    # so that finish cannot race an unacknowledged analysis/remaining worker.
    for path in sorted((root / "jobs").glob("*-submission.json")):
        if path.name not in ("cpu-submission.json", "prepare-submission.json"):
            analyses.append(read(path)["job"])
    completion = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:15:00",
        "--job-name=fox13-history-finished",
        f"--output={root}/slurm/history-finished-%j.out",
        f"--error={root}/slurm/history-finished-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_review_v13.sbatch"),
        str(SOURCE),
        str(root),
        "history-finished",
    ]
    try:
        history = importlib.import_module("evicastkv_history_v13")
        result = history.register(
            root / "history",
            cpu_gate_root=root,
            parent_campaign=PARENT,
            execute=True,
            completion_argv=completion,
            completion_dependencies=sorted(set(analyses)),
        )
        write_once(root / "history-submission.json", result)
        return dict(history=result, final="afterany_terminal_history_and_static_analyses")
    except Exception as error:
        shared.failure(root, "history-registration", error)
        if list((root / "history/jobs").glob("*-request.json")):
            # A request may already be accepted even if its acknowledgement is
            # lost. Do not submit a competing terminal callback or a duplicate
            # history worker. Preserve the exact request for reconciliation.
            return dict(
                status="blocked_or_unacknowledged_history_submission",
                automatic_retry=False,
                known_history_jobs=[
                    read(p)["job"]
                    for p in sorted((root / "history/jobs").glob("*-submission.json"))
                ],
            )
        return dict(final=submit(root, "final", "finish", dependencies=sorted(set(analyses))))


def analyze(root, lane):
    try:
        module(lane).analyze(root / lane)
    except Exception as error:
        shared.failure(root, f"analysis-{lane}", error)
    if lane == "ownership":
        try:
            tool = importlib.import_module("evicastkv_source_trace_adjudication")
            write_once(root / "ownership-review-inventory.json", tool.inventory(root / lane))
        except Exception as error:
            shared.failure(root, "ownership-review-inventory", error)
    return dict(lane=lane, gate_exists=(root / lane / LANES[lane][1]).exists())


def finish(root):
    plan = load(root)
    gates = {}
    for lane, (_, name) in LANES.items():
        path = root / lane / name
        gates[lane] = (
            dict(path=str(path), sha256=sha(path), evidence=read(path))
            if path.is_file()
            else dict(status="missing_gate", path=str(path))
        )
    history = root / "history/GATE_history_v13.json"
    gates["history"] = (
        dict(path=str(history), sha256=sha(history), evidence=read(history))
        if history.is_file()
        else dict(status="missing_gate", path=str(history))
    )
    errors = [
        dict(path=str(p), sha256=sha(p), evidence=read(p))
        for p in sorted((root / "errors").glob("*.json"))
    ]
    record = dict(
        source_sha=plan["source_sha"],
        status="completed_diagnostic_report_not_paper_ready",
        gates=gates,
        errors=errors,
        independent_quality_N=0,
        genuine_final_peak_byte_matching_established=False,
        fresh_confirmation_established=False,
        strict_latency_superiority_established=False,
        six_component_benefits_established=False,
        superiority_established=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "FINAL_report.json", record)
    lines = [
        "# V13 finite diagnostic results",
        "",
        f"Immutable source: `{plan['source_sha']}`.",
        "",
        "Each gate below is independent; successful execution is not superiority.",
        "",
        "| Lane | Gate |",
        "| --- | --- |",
    ]
    for lane, gate in gates.items():
        evidence = gate.get("evidence", gate)
        lines.append(f"| {lane} | {evidence.get('status', 'inspect explicit gate booleans')} |")
    lines.extend(
        [
            "",
            f"Recorded execution/analysis errors: {len(errors)}.",
            "",
            "Independent quality N = 0. Fresh confirmation, final/peak byte matching,",
            "strict latency superiority and six-component benefits are not established.",
            "No paper-scale campaign was automatically submitted.",
            "",
            "See FINAL_report.json and each immutable producer for complete observations.",
        ]
    )
    with (root / "SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return record


def dispatch(root, mode):
    load(root)
    if mode == "prepare":
        return prepare(root)
    if mode in {"finish", "history-finished"}:
        return finish(root)
    for lane in LANES:
        if mode == f"{lane}-worker":
            if lane != "providers" and shared.gpu_snapshot()["active_gpus"] > 6:
                raise ValueError("Aggregate six-A100 ceiling exceeded")
            return module(lane).worker(root / lane, int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
        if mode == f"after-{lane}":
            return analyze(root, lane)
    raise ValueError("Unknown finite V13 callback mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else dispatch(args.root, args.mode)
    )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
