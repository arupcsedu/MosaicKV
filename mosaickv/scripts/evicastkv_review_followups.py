#!/usr/bin/env python3
"""Finite, fail-closed consumed-input review callbacks; dry-run by default.

No polling, retries, optimization, configuration freeze or fresh confirmation.
Every model worker and failure is followed by one afterany CPU analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from evicastkv_campaign import admitted_array_concurrency

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
MODES = (
    "register",
    "cpu-audits",
    "after-mechanisms",
    "after-fullgraph",
    "graph-worker",
    "after-graph",
    "window-worker",
    "after-window",
    "boundary-worker",
    "after-boundary",
)
NO_CLAIMS = dict(
    fresh_confirmation_authorized=False,
    configuration_freeze_authorized=False,
    algorithm_optimization_authorized=False,
    superiority_established=False,
    whole_request_KV_peak_certified=False,
    prototype_residual_repair_claim_allowed=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def scratch(path):
    path = Path(path)
    if not path.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("all inputs and outputs must be under scratch")
    return path


def parent_controls(parent):
    lock_path = parent / "protocol_lock.json"
    lock = read(lock_path)
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    bindings = {str(lock_path): sha(lock_path)}
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        path = parent / f"analysis/{stage}/GATE_{stage}.json"
        gate = read(path)
        if (
            gate.get("status") != "passed"
            or gate.get("source_sha") != lock["source_sha"]
            or gate.get("protocol_sha256") != sha(lock_path)
            or gate.get("blockers")
        ):
            raise ValueError(f"parent {stage} gate is not passed and hash-bound")
        bindings[str(path)] = sha(path)
    return lock, bindings


def gpu_snapshot():
    """One event-time observation, including pending admissions; never a loop."""
    command = [
        "squeue",
        "-r",
        "-h",
        "-u",
        os.environ["USER"],
        "-t",
        "RUNNING,COMPLETING,PENDING",
        "-o",
        "%i|%b|%T",
    ]
    raw = subprocess.check_output(command, text=True)
    active, pending = [], []
    for line in raw.splitlines():
        identity, gres, state = line.strip().split("|", 2)
        matches = re.findall(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", gres)
        if "gpu" in gres and not matches:
            raise ValueError("unrecognized GPU ownership; refusing admission")
        if matches:
            item = dict(job=identity, gpus=sum(map(int, matches)), gres=gres, state=state)
            (pending if state == "PENDING" else active).append(item)
    return dict(
        command=command, active=active, pending=pending, active_gpus=sum(r["gpus"] for r in active)
    )


def admission(snapshot, desired):
    if desired not in {1, 6}:
        raise ValueError("only the one-worker graph or six-worker window scope is registered")
    limit = admitted_array_concurrency(
        dict(maximum_a100s=6, array_concurrency=desired), snapshot["active_gpus"]
    )
    # Wait for existing pending jobs too: their later allocation must not race
    # this finite controller. No other user's jobs are changed or depended on.
    dependencies = sorted(
        {r["job"].split("_", 1)[0] for r in snapshot["active"] + snapshot["pending"]}
    )
    if dependencies:
        limit = desired
    if limit < 1:
        raise ValueError("six-GPU ceiling has no admissible slot or completion dependency")
    return dict(
        snapshot=snapshot,
        maximum_a100s=6,
        requested_array_concurrency=desired,
        admitted_array_concurrency=limit,
        wait_for_existing_own_gpu_jobs=dependencies,
        prior_four_gpu_default_overridden_by_explicit_six_gpu_authorization=True,
        worker_entry_ceiling_check=True,
        concurrent_unrelated_new_submissions_not_authorized_by_this_controller=True,
    )


def submit(root, label, mode, dependencies=(), *, array=None, spec_sha=None, gpu=False):
    request = root / "jobs" / f"{label}-request.json"
    if request.exists():
        raise FileExistsError("finite callback submission already attempted; no in-place retry")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox-{label}",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if gpu:
        command += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=02:00:00",
            f"--array={array}",
        ]
    else:
        command += ["--partition=bii", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00"]
    if dependencies:
        if any(not re.fullmatch(r"\d+(?:_\d+)?", str(job)) for job in dependencies):
            raise ValueError("dependencies must name concrete scheduler jobs")
        command += ["--dependency=afterany:" + ":".join(map(str, dependencies))]
    command += [
        str(SOURCE / "mosaickv/slurm/evicastkv_review_followups.sbatch"),
        str(SOURCE),
        str(root),
        mode,
    ]
    if spec_sha is not None:
        command += ["--spec-sha", spec_sha]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=command, mode=mode, immutable_single_attempt=True))
    job = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", job):
        raise ValueError("scheduler returned an invalid job identifier")
    write_once(root / "jobs" / f"{label}-submission.json", dict(job=job, command=command))
    return job


def final_report(root, status, reason, **details):
    registration = read(root / "registration.json")
    derived = set(root.glob("GATE_*.json")) | set(root.glob("*/GATE_*.json"))
    derived.update(root.glob("*/analysis/*/GATE_*.json"))
    record = dict(
        status=status,
        reason=reason,
        source_sha=registration["source_sha"],
        registration_sha256=sha(root / "registration.json"),
        historical_bindings=registration["bindings"],
        derived_gate_bindings={str(path): sha(path) for path in sorted(derived)},
        consumed_development_only=True,
        original_artifacts_preserved=True,
        **NO_CLAIMS,
        **details,
    )
    write_once(root / "FINAL_report.json", record)
    path = root / "FINAL_report.md"
    with path.open("x") as stream:
        stream.write(
            f"# Prof. Fox review follow-up execution\n\nStatus: {status}.\n\n{reason}\n\n"
            "This finite consumed-input study does not establish superiority, fresh confirmation, "
            "a frozen candidate, serving support, prototype/repair performance, or whole-request "
            "KV-only peak certification. Original negative gates and raw records "
            "remain unchanged.\n"
        )
    return record


def register(root, parent, mechanisms, mile_scoring, *, execute=False):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("follow-up registration root must be new")
    _, bindings = parent_controls(parent)
    mechanism_lock = read(mechanisms / "protocol_lock.json")
    if mechanism_lock.get("parent_protocol_sha256") != sha(parent / "protocol_lock.json"):
        raise ValueError("mechanism study does not inherit the reviewed parent")
    clean_source(Path(mechanism_lock["source_root"]), mechanism_lock["source_sha"])
    mechanism_submission = mechanisms / "jobs/mechanisms-submission.json"
    analyzer = read(mechanism_submission)["analysis"]
    if not re.fullmatch(r"\d+", str(analyzer)):
        raise ValueError("mechanism analyzer job must be concrete")
    for path in (
        mechanisms / "protocol_lock.json",
        mechanism_submission,
        mile_scoring / "GATE_scoring.json",
        mile_scoring / "registration.json",
        mile_scoring / "paired_statistics.json",
    ):
        bindings[str(path)] = sha(path)
    for name in (
        "evicastkv_review_followups.py",
        "evicastkv_caption_audit.py",
        "evicastkv_transport_scores.py",
        "evicastkv_graph_profile.py",
        "evicastkv_window_study.py",
        "evicastkv_mile_parser_audit.py",
    ):
        path = SOURCE / "mosaickv/scripts" / name
        if not path.is_file():
            raise ValueError(f"required bounded follow-up implementation missing: {name}")
        bindings[str(path)] = sha(path)
    wrapper = SOURCE / "mosaickv/slurm/evicastkv_review_followups.sbatch"
    bindings[str(wrapper)] = sha(wrapper)
    boundary_module = SOURCE / "mosaickv/scripts/evicastkv_boundary_operators.py"
    boundary_registered = boundary_module.is_file()
    if boundary_registered:
        bindings[str(boundary_module)] = sha(boundary_module)
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            parent=str(parent),
            mechanisms=str(mechanisms),
            mechanism_analyzer_job=str(analyzer),
            mile_scoring=str(mile_scoring),
            bindings=bindings,
            execution_authorized=execute,
            dry_run=not execute,
            maximum_a100s=6,
            cpus_per_gpu_worker=12,
            wall_minutes=120,
            six_a100s_explicitly_authorized=True,
            full_graph_pilot_is_admission_dependency_not_original_policy_control=True,
            graph_profile_uses_original_parent_routing_not_fullroute_pilot=True,
            window_graph_diagnostic_pass_required=True,
            boundary_operator_diagnostic_registered=boundary_registered,
            boundary_operator_diagnostic_scope=(
                "two consumed COCO inputs; FullKV only; no quality or threshold change"
                if boundary_registered
                else "READY_NOT_RUN_implementation_not_registered"
            ),
            finite_stages=[
                "full_cpu_suite",
                "CPU_caption_quality_audits",
                "mechanisms_wait",
                "full_graph_wait",
                "one_graph_profile",
                "window_sensitivity",
            ],
            **NO_CLAIMS,
        ),
    )
    if not execute:
        return final_report(
            root, "DRY_RUN_ONLY", "No jobs submitted; use a new registration root with --execute."
        )
    # The existing compiler-pinned launcher executes the entire initialized CPU
    # unit suite and writes both its immutable JUnit and full log.
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=12G",
        "--time=00:10:00",
        "--job-name=fox-followup-cpu",
        f"--output={root}/slurm/full-cpu-%j.out",
        f"--error={root}/slurm/full-cpu-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_review_closure_v2.sbatch"),
        str(SOURCE),
        str(root / "full-cpu"),
        "tests",
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / "jobs/full-cpu-request.json", dict(command=command))
    cpu = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", cpu):
        raise ValueError("invalid CPU job identifier")
    write_once(root / "jobs/full-cpu-submission.json", dict(job=cpu, command=command))
    audits = submit(root, "cpu-audits", "cpu-audits", [cpu])
    callback = submit(root, "after-mechanisms", "after-mechanisms", [audits, analyzer])
    return dict(status="submitted_finite_chain", cpu=cpu, audits=audits, after_mechanisms=callback)


def load(root):
    registration = read(root / "registration.json")
    clean_source(SOURCE, registration["source_sha"])
    if not registration["execution_authorized"] or registration["dry_run"]:
        raise ValueError("dry-run registration does not authorize execution")
    for path, expected in registration["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("registered source or historical evidence changed")
    return registration


def cpu_gate(root, revision):
    path = root / "full-cpu/cpu-tests.json"
    record = read(path)
    if (
        record["status"] != "passed"
        or record["returncode"] != 0
        or record["source_sha"] != revision
    ):
        raise ValueError("complete current-source CPU suite did not pass")
    for name, key in (("cpu-tests.log", "log_sha256"), ("cpu-tests.xml", "junit_sha256")):
        if sha(root / "full-cpu" / name) != record[key]:
            raise ValueError("CPU suite log or JUnit changed")
    tree = ET.parse(root / "full-cpu/cpu-tests.xml").getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
    counts = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] < 1 or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("complete CPU suite includes missing, failed or skipped tests")
    return dict(status="passed", source_sha=revision, counts=counts, cpu_suite_sha256=sha(path))


def require_gate(path, *, protocol=None, status="passed"):
    gate = read(path)
    if gate.get("status") != status or gate.get("blockers"):
        raise ValueError(f"required empirical gate did not pass: {path}")
    if protocol is not None and gate.get("protocol_sha256") != sha(protocol):
        raise ValueError("empirical gate belongs to a different protocol")
    return gate


def require_cpu_audits(root):
    gate = require_gate(root / "GATE_cpu_audits.json")
    for path, expected in gate["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("derived CPU-audit gate changed")
    return gate


def run(root, mode, *, index=0, spec_sha=None):
    registration = load(root)
    parent, mechanisms = Path(registration["parent"]), Path(registration["mechanisms"])
    if mode == "cpu-audits":
        from evicastkv_caption_audit import audit
        from evicastkv_mile_parser_audit import audit as parser_audit
        from evicastkv_transport_scores import reexport

        gate = cpu_gate(root, registration["source_sha"])
        write_once(root / "GATE_cpu_suite.json", gate)
        audit(parent, root / "captions")
        reexport(Path(registration["mile_scoring"]), root / "mile-quality")
        parser_audit(Path(registration["mile_scoring"]), root / "mile-parser")
        gates = (
            root / "GATE_cpu_suite.json",
            root / "captions/GATE_caption_audit.json",
            root / "mile-quality/GATE_quality_statistics.json",
            root / "mile-parser/GATE_parser_audit.json",
        )
        write_once(
            root / "GATE_cpu_audits.json",
            dict(
                status="passed",
                source_sha=registration["source_sha"],
                new_GPU_inference=0,
                bindings={str(path): sha(path) for path in gates},
            ),
        )
        return
    if mode == "after-mechanisms":
        require_cpu_audits(root)
        path = mechanisms / "analysis/mechanisms/GATE_mechanisms.json"
        require_gate(path, protocol=mechanisms / "protocol_lock.json")
        submission = mechanisms / "jobs/full_graph_pilot-submission.json"
        analyzer = read(submission)["analysis"]
        write_once(
            root / "decisions/mechanisms.json",
            dict(
                status="passed", gate_sha256=sha(path), fullgraph_submission_sha256=sha(submission)
            ),
        )
        submit(root, "after-fullgraph", "after-fullgraph", [analyzer])
        return
    if mode == "after-fullgraph":
        # The new profile executes the parent's original r1/256 policy, not the
        # full-routing pilot. A negative full-routing result is retained, never
        # relabeled or inherited as a passing prerequisite for another policy.
        require_cpu_audits(root)
        require_gate(
            mechanisms / "analysis/mechanisms/GATE_mechanisms.json",
            protocol=mechanisms / "protocol_lock.json",
        )
        pilot = mechanisms / "analysis/full_graph_pilot/GATE_full_graph_pilot.json"
        pilot_record = read(pilot) if pilot.exists() else dict(status="missing_failed_analysis")
        write_once(
            root / "decisions/fullgraph.json",
            dict(
                status="dependency_finished_original_r256_controls_unchanged",
                full_graph_pilot_status=pilot_record.get("status"),
                full_graph_pilot_gate_sha256=sha(pilot) if pilot.exists() else None,
                full_graph_pilot_failure_not_reclassified=True,
                inherited_as_original_policy_control=False,
            ),
        )
        from evicastkv_graph_profile import STAGE, prepare

        prepare(root / "graph-profile", parent)
        spec = root / "graph-profile/jobs" / f"{STAGE}.json"
        admit = admission(gpu_snapshot(), 1)
        write_once(root / "admissions/graph.json", admit)
        gpu = submit(
            root,
            "graph",
            "graph-worker",
            admit["wait_for_existing_own_gpu_jobs"],
            array="0%1",
            spec_sha=sha(spec),
            gpu=True,
        )
        submit(root, "after-graph", "after-graph", [gpu])
        return
    if mode in {"graph-worker", "window-worker", "boundary-worker"}:
        snapshot = gpu_snapshot()
        if snapshot["active_gpus"] > 6:
            raise RuntimeError("aggregate six-A100 ceiling exceeded; no model work started")
        write_once(root / f"admissions/{mode}-{index}.json", snapshot)
        if mode == "graph-worker":
            from evicastkv_graph_profile import worker

            worker(root / "graph-profile", index, spec_sha)
        elif mode == "window-worker":
            from evicastkv_window_study import worker

            worker(root / "windows", index, spec_sha)
        else:
            from evicastkv_boundary_operators import worker

            if not registration["boundary_operator_diagnostic_registered"] or index != 0:
                raise ValueError("unregistered independent Boundary-B diagnostic")
            worker(root / "boundary-operators")
        return
    if mode == "after-graph":
        from evicastkv_graph_profile import analyze

        analyze(root / "graph-profile")
        require_gate(
            root / "graph-profile/analysis/graph_cpu_profile/GATE_graph_cpu_profile.json",
            protocol=root / "graph-profile/protocol_lock.json",
        )
        from evicastkv_window_study import STAGE, prepare

        prepare(root / "windows", parent)
        spec = root / "windows/jobs" / f"{STAGE}.json"
        jobs = read(spec)["jobs"]
        if len(jobs) != 16 or any(len(job["indices"]) != 4 for job in jobs):
            raise ValueError("window study must retain its sixteen four-input shard plan")
        admit = admission(gpu_snapshot(), 6)
        write_once(root / "admissions/windows.json", admit)
        array = "0-15%" + str(admit["admitted_array_concurrency"])
        gpu = submit(
            root,
            "windows",
            "window-worker",
            admit["wait_for_existing_own_gpu_jobs"],
            array=array,
            spec_sha=sha(spec),
            gpu=True,
        )
        submit(root, "after-window", "after-window", [gpu])
        return
    if mode == "after-window":
        from evicastkv_window_study import STAGE, analyze

        analyze(root / "windows")
        path = root / f"windows/analysis/{STAGE}/GATE_{STAGE}.json"
        require_gate(
            path,
            protocol=root / "windows/protocol_lock.json",
            status="passed_exploratory_sensitivity_not_confirmation",
        )
        if registration["boundary_operator_diagnostic_registered"]:
            from evicastkv_boundary_operators import prepare

            write_once(
                root / "decisions/windows.json",
                dict(status="passed_exploratory_not_confirmation", window_gate_sha256=sha(path)),
            )
            # This independent FullKV numerical diagnosis does not select a
            # candidate, change a threshold, or waive the negative B controls.
            prepare(parent, root / "boundary-operators")
            admit = admission(gpu_snapshot(), 1)
            write_once(root / "admissions/boundary.json", admit)
            gpu = submit(
                root,
                "boundary",
                "boundary-worker",
                admit["wait_for_existing_own_gpu_jobs"],
                array="0%1",
                gpu=True,
            )
            submit(root, "after-boundary", "after-boundary", [gpu])
            return
        return final_report(
            root,
            "COMPLETED_CONSUMED_DIAGNOSTICS_ONLY",
            "CPU audits, original-policy graph profiling and registered prompt-window "
            "sensitivity completed. No configuration selected or independent campaign launched.",
            window_gate_sha256=sha(path),
            boundary_operator_diagnostic="READY_NOT_RUN_not_registered_in_immutable_source",
        )
    if mode == "after-boundary":
        if not registration["boundary_operator_diagnostic_registered"]:
            raise ValueError("Boundary-B diagnostic was not registered in the immutable source")
        from evicastkv_boundary_operators import analyze

        path = root / "boundary-operators/GATE_boundary_operators.json"
        error = None
        try:
            analyze(root / "boundary-operators")
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            write_once(
                root / "decisions/boundary-analysis-failure.json",
                dict(status="blocked_diagnostic_execution", error=error),
            )
        gate = read(path) if path.exists() else dict(status="blocked_diagnostic_execution")
        return final_report(
            root,
            "COMPLETED_CONSUMED_A_STUDIES_BOUNDARY_SEPARATE",
            "The exact-cache study remains separate from the independent FullKV "
            "Boundary-B numerical diagnosis. Negative B controls are preserved, "
            "not used to reverse the exact-cache study or authorize quality claims.",
            boundary_diagnostic_status=gate["status"],
            boundary_gate_sha256=sha(path) if path.exists() else None,
            boundary_analysis_error=error,
            fullkv_chunked_logit_control_all_passed=gate.get(
                "fullkv_chunked_logit_control_all_passed"
            ),
            boundary_quality_inference_authorized=False,
        )
    raise ValueError("unknown finite callback mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--mechanisms", type=Path)
    parser.add_argument("--mile-scoring", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    for path in (args.root, args.parent, args.mechanisms, args.mile_scoring):
        if path is not None:
            scratch(path)
    try:
        if args.mode == "register":
            if any(path is None for path in (args.parent, args.mechanisms, args.mile_scoring)):
                parser.error("register requires parent, mechanisms and mile-scoring")
            result = register(
                args.root, args.parent, args.mechanisms, args.mile_scoring, execute=args.execute
            )
        else:
            result = run(args.root, args.mode, index=args.index, spec_sha=args.spec_sha)
        print(json.dumps(result))
    except Exception as error:
        if (args.root / "registration.json").exists():
            label = f"{args.mode}-{args.index}" if args.mode.endswith("-worker") else args.mode
            write_once(
                args.root / f"decisions/{label}-failure.json",
                dict(status="blocked", error=f"{type(error).__name__}: {error}", **NO_CLAIMS),
            )
            if not args.mode.endswith("-worker") and not (args.root / "FINAL_report.json").exists():
                final_report(args.root, "BLOCKED", f"{args.mode}: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
