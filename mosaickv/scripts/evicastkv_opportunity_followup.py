#!/usr/bin/env python3
"""One finite, preregistered r16 follow-up; no polling and no self-modifying loop."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from evicastkv_campaign import admitted_array_concurrency, worker_time_limit
from evicastkv_opportunity_development import quality_decision
from evicastkv_opportunity_study import arms

from mosaickv.config import RunConfig, canonical_config
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
REGISTRATION = SOURCE / "mosaickv/configs/experiments/evicastkv_opportunity_followup_v1.yaml"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PATHS = ("graph", "prototypes")


def validate_registration(registration):
    if (
        registration["budgets_mib"] != [20, 40]
        or registration["canary_samples"] != 2
        or registration["development_samples"] != 64
        or registration["minimum_query_mass_fraction"] != 0.95
        or registration["shortlist_size"] != 8
        or registration["maximum_additional_gpu_requests"] != 858
        or registration["no_optional_enlargement"] is not True
        or registration["only_treatment_change"]
        != {
            "graph.uncertain_head_fraction": 1 / 16,
            "prototypes.uncertain_head_fraction": 1 / 16,
        }
        or registration["statistics"] != dict(resamples=10000, seed=20271014, family_size=18)
        or any(
            registration[k]
            for k in (
                "fresh_cohorts_authorized",
                "confirmation_authorized",
                "paper_campaign_authorized",
            )
        )
    ):
        raise ValueError("finite follow-up scope or thresholds changed")
    resource = registration["resources"]
    required = dict(
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        array_concurrency=3,
        cpus_per_worker=12,
        samples_per_worker=4,
        memory_gib_per_worker=64,
        partition="bii-gpu",
        reservation="bi_fox_dgx",
        account="bii_dsc_community",
    )
    if any(resource.get(k) != v for k, v in required.items()):
        raise ValueError("registered resource controls changed")
    worker_time_limit(resource)
    worker_time_limit(dict(wall_minutes=registration["canary_wall_minutes"]))


def r16_template(original):
    """Change only participation, never precision, quota, coefficients or floor."""
    revised = copy.deepcopy(original)
    for section in PATHS:
        if original[section]["uncertain_head_fraction"] != 1 / 256:
            raise ValueError("follow-up requires the registered 1/256 reference")
        revised[section]["uncertain_head_fraction"] = 1 / 16
    config = RunConfig.from_mapping(revised)
    if canonical_config(config).get("evidence_policy"):
        raise ValueError("template must not preinstall an evidence-policy arm")
    restored = copy.deepcopy(revised)
    for section in PATHS:
        restored[section]["uncertain_head_fraction"] = 1 / 256
    if restored != original:
        raise ValueError("unexpected non-routing treatment change")
    return revised


def next_action(gate):
    if gate.get("status") != "passed_execution_development_only" or gate.get("actual_rows") != 832:
        return "blocked_incomplete_or_invalid_development"
    expected = {"evidence-routing-v1", "evidence-coverage-v1", "evidence-both-v1"}
    decisions = gate.get("quality_decisions", {})
    if set(decisions) != expected or any(
        d.get("required_contrasts") != 4 for d in decisions.values()
    ):
        raise ValueError("development decision lacks the prespecified contrasts")
    if any(type(d.get("positive_development_signal")) is not bool for d in decisions.values()):
        raise ValueError("quality decisions must contain actual booleans")
    if any(d["positive_development_signal"] for d in decisions.values()):
        return "stop_development_signal_requires_independent_confirmation"
    return "run_registered_r16_numerical_canary"


def env(runtime):
    value = dict(os.environ)
    value["PYTHONPATH"] = f"{runtime}/mosaickv/src:{runtime}/mosaickv/scripts"
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    return value


def load(root):
    plan = json.loads((root / "plan.json").read_text())
    clean_source(SOURCE, plan["controller_sha"])
    clean_source(Path(plan["registration"]["runtime_source"]), plan["registration"]["runtime_sha"])
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("follow-up binding changed")
    return plan


def schedule(root, phase, dependency):
    path = root / "jobs" / f"{phase}.json"
    if path.exists():
        raise FileExistsError("completion callback already submitted")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--gres=none",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:05:00",
        f"--dependency=afterany:{dependency}",
        f"--output={root}/slurm/{phase}-%j.out",
        f"--error={root}/slurm/{phase}-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_opportunity_followup.sbatch"),
        str(SOURCE),
        str(root),
        phase,
    ]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(path, dict(job_id=job, dependency=dependency, command=command))
    return job


def register(root, *, first_root=None, analysis_job=None):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("follow-up root must be new")
    registration = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(registration)
    if first_root is not None:
        recovery = json.loads((first_root / "analysis_recovery.json").read_text())
        if (
            Path(recovery["producer_root"]).resolve() != Path(registration["first_root"]).resolve()
            or recovery["new_GPU_requests"] != 0
            or recovery["reused_rows"] != 832
            or sha(first_root / "protocol_lock.json") != recovery["original_protocol_sha256"]
        ):
            raise ValueError("replacement first root must be a CPU-only recovery of the original")
        registration["first_root"] = str(first_root)
        registration["first_analysis_job"] = analysis_job
    first = Path(registration["first_root"])
    lock_path = first / "protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    runtime = Path(registration["runtime_source"])
    clean_source(runtime, registration["runtime_sha"])
    if lock["source_sha"] != registration["runtime_sha"]:
        raise ValueError("first study source changed")
    submission = first / "jobs/submission.json"
    if json.loads(submission.read_text())["analysis"] != registration["first_analysis_job"]:
        raise ValueError("callback is not bound to the real first analysis")
    template = Path(lock["templates"]["coco"]["path"])
    if sha(template) != lock["templates"]["coco"]["sha256"]:
        raise ValueError("original template changed")
    r16_template(json.loads(template.read_text()))
    root.mkdir(parents=True)
    (root / "slurm").mkdir()
    write_once(
        root / "plan.json",
        dict(
            controller_sha=revision,
            registration=registration,
            bindings={
                str(REGISTRATION): sha(REGISTRATION),
                str(lock_path): sha(lock_path),
                str(submission): sha(submission),
                str(template): sha(template),
            },
            fresh_confirmation=False,
            no_automatic_paper_claim=True,
        ),
    )
    job = schedule(root, "after-first", registration["first_analysis_job"])
    print(json.dumps(dict(controller=job, dependent_on=registration["first_analysis_job"])))


def prepare_derived(root, plan, stage):
    if stage not in {"evidence_canary", "evidence_development"}:
        raise ValueError("unregistered derived stage")
    registration = plan["registration"]
    first = Path(registration["first_root"])
    lock = copy.deepcopy(json.loads((first / "protocol_lock.json").read_text()))
    is_canary = stage == "evidence_canary"
    target = root / ("r16-canary" if is_canary else "r16-development")
    if target.exists():
        raise FileExistsError("derived execution root must be new")
    gate = None
    if not is_canary:
        gate = root / "r16-canary/analysis/evidence_canary/GATE_evidence_canary.json"
        if json.loads(gate.read_text())["status"] != "passed_consumed_correctness_only":
            raise ValueError("r16 numerical canary did not pass")
    target.mkdir()
    template = r16_template(json.loads(Path(lock["templates"]["coco"]["path"]).read_text()))
    template_path = target / "templates/coco-r16.json"
    write_once(template_path, template)
    lock["templates"]["coco"] = dict(path=str(template_path), sha256=sha(template_path))
    if is_canary:
        original = lock["cohorts"]["coco"]["development"]
        ledger = json.loads(Path(original["path"]).read_text())
        ledger["samples"] = ledger["samples"][:2]
        path = target / "manifests/consumed-canary.json"
        write_once(path, ledger)
        lock["cohorts"] = {
            "coco": {
                "consumed_canary": dict(
                    path=str(path),
                    sha256=sha(path),
                    sample_ids=[s["sample_id"] for s in ledger["samples"]],
                )
            }
        }
    lock["resources"] = dict(registration["resources"])
    if is_canary:
        lock["resources"].update(
            array_concurrency=2, wall_minutes=registration["canary_wall_minutes"]
        )
    lock["name"] = f"evicastkv_opportunity_r16_{stage}"
    lock["stages"] = [stage]
    lock["routing_followup"] = dict(
        routing_fraction=1 / 16,
        original_routing_fraction=1 / 256,
        entropy_comparator_also_uses_r16=True,
        comparator_not_the_original_frozen_r256=True,
        classification="already_consumed_development_only",
        conditional_on_negative_first_development=True,
        controller_source_sha=plan["controller_sha"],
        no_budget_weight_floor_precision_or_generation_change=True,
    )
    lock["historical_bindings"].update(plan["bindings"])
    lock["historical_bindings"][str(root / "plan.json")] = sha(root / "plan.json")
    if gate is not None:
        lock["historical_bindings"][str(gate)] = sha(gate)
    write_once(target / "protocol_lock.json", lock)
    jobs = [
        dict(
            workload="coco",
            stage=stage,
            subset="consumed_canary" if is_canary else "development",
            indices=[i] if is_canary else list(range(i, i + 4)),
            arms=arms(),
        )
        for i in (range(2) if is_canary else range(0, 64, 4))
    ]
    write_once(
        target / f"jobs/{stage}.json",
        dict(
            stage=stage,
            protocol_sha256=sha(target / "protocol_lock.json"),
            jobs=jobs,
        ),
    )
    write_once(
        target / "readiness.json",
        dict(
            status="prepared_r16_consumed_canary" if is_canary else "prepared_r16_development_only",
            source_sha=lock["source_sha"],
            requests=26 if is_canary else 832,
            fresh_confirmation=False,
            routing_fraction=1 / 16,
            comparator="entropy_r16_not_the_original_frozen_r256",
        ),
    )
    return target, lock


def submit_derived(target, lock, stage):
    runtime = Path(lock["source_root"])
    active = subprocess.check_output(
        [
            "squeue",
            "-h",
            "-u",
            os.environ["USER"],
            "-t",
            "RUNNING,COMPLETING,PENDING",
            "-o",
            "%i|%b",
        ],
        text=True,
    )
    total = sum(
        int(m[1])
        for line in active.splitlines()
        if (m := re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", line))
    )
    concurrency = admitted_array_concurrency(lock["resources"], total)
    if concurrency < 1:
        raise ValueError("aggregate six-A100 admission reached")
    spec = target / f"jobs/{stage}.json"
    count = len(json.loads(spec.read_text())["jobs"])
    is_canary = stage == "evidence_canary"
    stem = "opportunity_canary" if is_canary else "opportunity_development"
    analysis_stem = "opportunity_analysis" if is_canary else "opportunity_development_analysis"
    (target / "slurm").mkdir()
    common = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    command = [
        *common,
        f"--array=0-{count - 1}%{concurrency}",
        f"--time={worker_time_limit(lock['resources'])}",
        f"--output={target}/slurm/gpu-%A_%a.out",
        f"--error={target}/slurm/gpu-%A_%a.err",
        str(runtime / f"mosaickv/slurm/evicastkv_{stem}.sbatch"),
        str(runtime),
        str(target),
        "worker",
        sha(spec),
    ]
    gpu = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(target / "jobs/gpu_submission.json", dict(array=gpu, command=command))
    analysis_source = runtime if is_canary else SOURCE
    callback = [
        *common,
        "--partition=bii",
        "--gres=none",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:10:00",
        f"--dependency=afterany:{gpu}",
        f"--output={target}/slurm/analysis-%j.out",
        f"--error={target}/slurm/analysis-%j.err",
        str(analysis_source / f"mosaickv/slurm/evicastkv_{analysis_stem}.sbatch"),
        str(analysis_source),
        str(target),
    ]
    cpu = subprocess.check_output(callback, text=True).strip().split(";")[0]
    write_once(target / "jobs/analysis_submission.json", dict(analysis=cpu, command=callback))
    write_once(
        target / "jobs/submission.json",
        dict(
            array=gpu,
            analysis=cpu,
            command=command,
            callback=callback,
        ),
    )
    return cpu


def execute(root, phase):
    plan = load(root)
    decision_path = root / "decisions" / f"{phase}.json"
    if decision_path.exists():
        raise FileExistsError("completion decision is write-once")
    registration = plan["registration"]
    first = Path(registration["first_root"])
    if phase == "after-first":
        gate_path = first / "analysis/evidence_development/GATE_evidence_development.json"
        gate = json.loads(gate_path.read_text())
        if gate.get("status") == "passed_execution_development_only" and (
            gate.get("source_sha") != registration["runtime_sha"]
            or gate.get("protocol_sha256") != sha(first / "protocol_lock.json")
        ):
            raise ValueError("first gate source/protocol changed")
        action = next_action(gate)
        bindings = {str(gate_path): sha(gate_path)}
        if action == "run_registered_r16_numerical_canary":
            stats_path = gate_path.parent / "quality_statistics.json"
            if quality_decision(json.loads(stats_path.read_text())) != gate["quality_decisions"]:
                raise ValueError("gate decisions disagree with the actual quality statistics")
            target, lock = prepare_derived(root, plan, "evidence_canary")
            dependency = submit_derived(target, lock, "evidence_canary")
            callback = schedule(root, "after-canary", dependency)
            action += f"; callback={callback}"
    elif phase == "after-canary":
        gate_path = root / "r16-canary/analysis/evidence_canary/GATE_evidence_canary.json"
        gate = json.loads(gate_path.read_text())
        bindings = {str(gate_path): sha(gate_path)}
        action = "blocked_failed_r16_numerical_canary"
        if gate.get("status") == "passed_consumed_correctness_only" and not gate.get("errors"):
            if (
                gate.get("actual_rows") != 26
                or gate.get("paired_budgets") != 4
                or gate.get("source_sha") != registration["runtime_sha"]
                or gate.get("protocol_sha256") != sha(root / "r16-canary/protocol_lock.json")
            ):
                raise ValueError("r16 canary is not complete or source/protocol changed")
            target, lock = prepare_derived(root, plan, "evidence_development")
            dependency = submit_derived(target, lock, "evidence_development")
            callback = schedule(root, "after-second", dependency)
            action = f"run_registered_r16_development; callback={callback}"
    elif phase == "after-second":
        gate_path = (
            root / "r16-development/analysis/evidence_development/GATE_evidence_development.json"
        )
        gate = json.loads(gate_path.read_text())
        bindings = {str(gate_path): sha(gate_path)}
        if gate.get("status") == "passed_execution_development_only" and (
            gate.get("source_sha") != registration["runtime_sha"]
            or gate.get("protocol_sha256") != sha(root / "r16-development/protocol_lock.json")
        ):
            raise ValueError("second gate source/protocol changed")
        action = next_action(gate)
        if action == "run_registered_r16_numerical_canary":
            action = "stop_negative_or_inconclusive_r16_development_no_more_registered_gpu_runs"
    else:
        raise ValueError("unknown finite follow-up phase")
    write_once(
        decision_path,
        dict(
            phase=phase,
            action=action,
            bindings=bindings,
            controller_source_sha=plan["controller_sha"],
            no_confirmation_or_paper_claim=True,
            no_fresh_ids=True,
        ),
    )
    print(json.dumps(dict(phase=phase, action=action)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "after-first", "after-canary", "after-second"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--first-root", type=Path)
    parser.add_argument("--analysis-job")
    a = parser.parse_args()
    if not a.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only follow-up artifacts")
    if a.mode == "register":
        register(a.root, first_root=a.first_root, analysis_job=a.analysis_job)
    else:
        try:
            execute(a.root, a.mode)
        except Exception as error:
            failure = a.root / "decisions" / f"{a.mode}.json"
            if not failure.exists():
                write_once(
                    failure,
                    dict(
                        phase=a.mode,
                        action="blocked_controller_error",
                        error=f"{type(error).__name__}: {error}",
                        no_confirmation_or_paper_claim=True,
                    ),
                )
            raise
