#!/usr/bin/env python3
"""Finite post-gate CPU continuation; never GPU work or a fresh cohort."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path

import evicastkv_coco_boundary_v28 as numerical
import evicastkv_coco_history_subset as history
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SCRATCH = Path("/scratch/djy8hg")
POLICY = dict(
    id="V30_gated_finite_CPU_history_continuation_v1",
    original_worker_indices=[1, 7],
    selected_directories=521,
    worker_seconds=210,
    worker_bytes=2 * 2**30,
    CPU_suite_timeout_seconds=840,
    GPU_jobs=0,
    new_identity_selection_authorized=False,
    materialization_authorized=False,
    fresh_confirmation_authorized=False,
    global_freshness_certification=False,
    automatic_retry=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def empirical_bindings(whole_path, coco_path, revision):
    """Verify engineering admission and immutable proof files before queuing."""
    whole_path, coco_path = (Path(path).resolve(strict=True) for path in (whole_path, coco_path))
    for path in (whole_path, coco_path):
        path.relative_to(SCRATCH.resolve())
    whole, coco = read(whole_path), read(coco_path)
    bindings = numerical.require_whole_gate(whole, revision)
    history.empirical_controls(
        whole, coco, whole_producer_sha=whole["source_sha"], coco_producer_sha=revision
    )
    if len(coco.get("reports", [])) != 2 or any(
        report.get("sample_id") != sample
        or report.get("predictions") != 128
        or report.get("prediction_agreement") != 128
        or report.get("pass_128") is not True
        for report, sample in zip(coco["reports"], numerical.CONSUMED_IDS, strict=True)
    ):
        raise ValueError("two complete original COCO 128-prediction controls required")
    paths = [whole_path, coco_path]
    if whole.get("gate_origin") == numerical.POLICY["derived_prerequisite_origin"]:
        provenance_path = whole_path.parent / "provenance.json"
        provenance = read(provenance_path)
        if (
            provenance.get("execution_source_sha") != whole["source_sha"]
            or provenance.get("analyzer_source_sha") != revision
        ):
            raise ValueError("whole-request execution/validation provenance differs")
        if (
            provenance.get("GPU_inference_repeated") is not False
            or provenance.get("preserves_original_artifacts") is not True
        ):
            raise ValueError("derived validation cannot masquerade as new inference")
        bindings.update(provenance["bindings"])
        paths.append(provenance_path)
    coco_root = coco_path.parent.parent
    plan_path = coco_root / "plan.json"
    if sha(plan_path) != coco["plan_sha256"] or read(plan_path)["source_sha"] != revision:
        raise ValueError("COCO control plan/source binding differs")
    paths.append(plan_path)
    for index in (0, 1):
        worker = coco_root / f"workers/{index}"
        result = read(worker / "result.json")
        if (
            result.get("source_sha") != revision
            or result.get("failures") != []
            or result.get("plan_sha256") != coco["plan_sha256"]
        ):
            raise ValueError("COCO worker provenance/completeness differs")
        paths.append(worker / "result.json")
        for name, expected in result["outputs"].items():
            path = (worker / name).resolve(strict=True)
            if not path.is_relative_to(worker.resolve()) or sha(path) != expected:
                raise ValueError("COCO raw numerical evidence changed")
            bindings[str(path)] = expected
    bindings.update({str(path): sha(path) for path in paths})
    for name, expected in bindings.items():
        path = Path(name).resolve(strict=True)
        path.relative_to(SCRATCH.resolve())
        if sha(path) != expected:
            raise ValueError("empirical admission artifact changed")
    return bindings, whole["source_sha"]


def load(root):
    root = Path(root)
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or plan["policy_sha256"] != digest(POLICY):
        raise ValueError("immutable CPU continuation policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable CPU continuation input changed")
    history.empirical_controls(
        read(plan["whole_gate"]),
        read(plan["coco_gate"]),
        whole_producer_sha=plan["whole_producer_sha"],
        coco_producer_sha=plan["source_sha"],
    )
    return plan


def submit(root, revision, label, mode, *, dependency=None):
    root = Path(root)
    worker = mode == "worker"
    analysis = mode == "analyze"
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii,standard,interactive",
        f"--job-name=fox30-{label}",
        f"--cpus-per-task={12 if worker or analysis else 4}",
        f"--mem={'16G' if analysis else '8G'}",
        f"--time=00:{15 if mode == 'tests' else 5}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if worker:
        command.append("--array=1,7%2")
    if dependency is not None:
        if not str(dependency).isdigit():
            raise ValueError("numeric acknowledged dependency required")
        command.append(f"--dependency=afterany:{dependency}")
    command += [
        str(SOURCE / "mosaickv/slurm/fox_review_cpu_continuation_v30.sbatch"),
        str(SOURCE),
        str(root),
        mode,
        revision,
    ]
    write_once(root / f"jobs/{label}-request.json", dict(command=command, source_sha=revision))
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("uncertain CPU scheduler acknowledgement; do not retry")
    write_once(
        root / f"jobs/{label}-submission.json", dict(command=command, source_sha=revision, job=job)
    )
    return job


def saved_media_bindings(root, revision):
    from mosaickv.evaluation.coco_hash_recovery import DATASET_REVISION, PRODUCER

    root = Path(root).resolve(strict=True)
    root.relative_to(SCRATCH.resolve())
    gate_path, registration_path = (
        root / "GATE_saved_COCO_hash_recovery.json",
        root / "registration.json",
    )
    record, registration = read(gate_path), read(registration_path)
    if (
        record.get("reader_source_sha") != revision
        or registration.get("reader_source_sha") != revision
        or record.get("original_producer_sha") != PRODUCER
        or registration.get("original_producer_sha") != PRODUCER
        or record.get("dataset_revision") != DATASET_REVISION
        or record.get("complete_saved_hash_reduction") is not True
        or record.get("hash_rows") != 40504
        or not isinstance(registration.get("inputs"), dict)
        or not registration["inputs"]
    ):
        raise ValueError("only complete pinned current-source saved hash reduction may be reused")
    for name in (
        "globally_fresh",
        "fresh_confirmation_authorized",
        "materialization_authorized",
        "candidate_ids_selected_or_exported",
    ):
        if record.get(name) is not False:
            raise ValueError("saved hashes cannot certify freshness or select identities")
    bindings = {str(path): sha(path) for path in (gate_path, registration_path)}
    for item in registration["inputs"].values():
        path = Path(item["path"]).resolve(strict=True)
        path.relative_to(SCRATCH.resolve())
        if sha(path) != item["sha256"]:
            raise ValueError("saved hash inputs changed")
        bindings[str(path)] = item["sha256"]
    return bindings


def launch(root, whole_gate, coco_gate, *, cpu_gate_root=None, media_recovery_root=None):
    revision = clean_source(SOURCE)
    bindings, whole_producer = empirical_bindings(whole_gate, coco_gate, revision)
    root = Path(root).resolve()
    scratch = (SCRATCH / "workdir").resolve()
    if root == scratch or not root.is_relative_to(scratch) or root.exists():
        raise ValueError("new immutable scratch CPU continuation root required")
    reuse = False
    if cpu_gate_root is not None:
        Path(cpu_gate_root).resolve().relative_to(SCRATCH.resolve())
        record_path = Path(cpu_gate_root) / "full-cpu/cpu-tests.json"
        if record_path.exists() and read(record_path).get("source_sha") == revision:
            cpu_gate(Path(cpu_gate_root), revision)
            reuse = True
            for name in ("cpu-tests.json", "cpu-tests.log", "cpu-tests.xml"):
                path = Path(cpu_gate_root) / "full-cpu" / name
                bindings[str(path.resolve())] = sha(path)
    if media_recovery_root is not None:
        Path(media_recovery_root).resolve(strict=True).relative_to(SCRATCH.resolve())
        bindings.update(saved_media_bindings(media_recovery_root, revision))
    cpu_root = Path(cpu_gate_root).resolve() if reuse else root
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "slurm").mkdir()
    write_once(
        root / "plan.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            bindings=bindings,
            whole_gate=str(Path(whole_gate).resolve()),
            coco_gate=str(Path(coco_gate).resolve()),
            whole_producer_sha=whole_producer,
            cpu_gate_root=str(cpu_root),
            reuse_passed_current_CPU=reuse,
            media_recovery_root=str(Path(media_recovery_root).resolve())
            if media_recovery_root
            else None,
        ),
    )
    tests = None if reuse else submit(root, revision, "cpu", "tests")
    callback = submit(root, revision, "after-cpu", "after-cpu", dependency=tests)
    return dict(cpu=tests, after_cpu=callback, GPU_jobs=0)


def after_cpu(root):
    root = Path(root)
    revision = read(root / "plan.json")["source_sha"]
    errors = []
    workers = None
    try:
        plan = load(root)
        cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
        history.register(
            root / "subset",
            execute=True,
            cpu_gate_root=Path(plan["cpu_gate_root"]),
            whole_gate=Path(plan["whole_gate"]),
            coco_gate=Path(plan["coco_gate"]),
            whole_producer_sha=plan["whole_producer_sha"],
            coco_producer_sha=plan["source_sha"],
        )
        workers = submit(root, plan["source_sha"], "history", "worker")
    except Exception as exc:
        errors.append(
            dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        )
    write_once(
        root / "decisions/after-cpu.json",
        dict(admitted=not errors, failures=errors, workers=workers),
    )
    analysis = submit(root, revision, "analysis", "analyze", dependency=workers)
    return dict(workers=workers, analysis=analysis, GPU_jobs=0)


def recover_saved_media(plan, root):
    from mosaickv.evaluation.coco_hash_recovery import recover

    reuse = plan["media_recovery_root"]
    if reuse:
        path = Path(reuse) / "GATE_saved_COCO_hash_recovery.json"
        saved_media_bindings(reuse, plan["source_sha"])
        return dict(reused=True, path=str(path), sha256=sha(path))
    record = recover(root / "saved-media", reader_source_sha=plan["source_sha"])
    return dict(
        reused=False,
        path=str(root / "saved-media/GATE_saved_COCO_hash_recovery.json"),
        complete_saved_hash_reduction=record["complete_saved_hash_reduction"],
    )


def analyze(root):
    root = Path(root)
    raw_plan = read(root / "plan.json")
    subset = media = None
    try:
        plan = load(root)
        decision = read(root / "decisions/after-cpu.json")
        errors = list(decision["failures"])
    except Exception as exc:
        plan = raw_plan
        decision = dict(admitted=False)
        errors = [
            dict(
                stage="admission_or_integrity",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        ]
    if decision["admitted"] is True:
        for name, action in (
            ("subset", lambda: history.analyze(root / "subset")),
            ("saved-media", lambda: recover_saved_media(plan, root)),
        ):
            try:
                result = action()
                if name == "subset":
                    subset = result
                    if result["missing_original_worker_indices"] or result["new_coverage_errors"]:
                        errors.append(
                            dict(
                                stage="subset_completeness",
                                missing=result["missing_original_worker_indices"],
                                new_coverage_errors=result["new_coverage_errors"],
                            )
                        )
                else:
                    media = result
            except Exception as exc:
                errors.append(
                    dict(
                        stage=name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )
    error_files = {
        str(path): read(path) for path in sorted((root / "worker-errors").glob("*.json"))
    }
    write_once(
        root / "FINAL_CPU_continuation.json",
        dict(
            source_sha=plan["source_sha"],
            status="completed_bounded_metadata_work"
            if not errors and not error_files
            else "blocked_or_partial_metadata_work",
            failures=errors,
            worker_errors=error_files,
            subset_report=subset,
            saved_media=media,
            GPU_jobs_submitted=0,
            new_identity_selection_authorized=False,
            globally_fresh=False,
            materialization_authorized=False,
            fresh_confirmation_authorized=False,
            large_campaign_authorized=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "after-cpu", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--whole-gate", type=Path)
    parser.add_argument("--coco-gate", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--media-recovery-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "launch":
        if args.whole_gate is None or args.coco_gate is None:
            parser.error("both empirical gates required")
        result = launch(
            args.root,
            args.whole_gate,
            args.coco_gate,
            cpu_gate_root=args.cpu_gate_root,
            media_recovery_root=args.media_recovery_root,
        )
    elif args.mode == "after-cpu":
        result = after_cpu(args.root)
    elif args.mode == "analyze":
        result = analyze(args.root)
    else:
        if args.index not in (1, 7):
            parser.error("retain original indices1and7")
        load(args.root)
        try:
            result = history.worker(args.root / "subset", args.index)
        except Exception as exc:
            write_once(
                args.root / f"worker-errors/{args.index}.json",
                dict(error=str(exc), traceback=traceback.format_exc()),
            )
            raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
