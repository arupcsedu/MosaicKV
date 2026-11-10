#!/usr/bin/env python3
"""One CPU analysis correction, then the already prepared consumed BF16 bridge.

No source inference retry, fresh cohort, tolerance change, or polling loop.
Original errors/negative controls stay immutable and remain in the final packet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from evicastkv_review_followups import cpu_gate, gpu_snapshot

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110")
OLD_SHA = "de339c2a66ed9804a1348ae0767f27588074983a"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    value = str(value).strip()
    if re.fullmatch(r"\d+(?:;[A-Za-z0-9_-]+)?", value) is None:
        raise ValueError("concrete scheduler ID required")
    return value.split(";", 1)[0]


def safe_root(root):
    root = Path(root)
    base = Path("/scratch/djy8hg/audits").resolve()
    root.resolve().relative_to(base)
    if root.resolve() == base:
        raise ValueError("dedicated recovery directory required")
    return root


def submit(root, label, mode, dependencies=(), *, tests=False, bridge=False):
    registration = read(root / "registration.json")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox7r-{label}",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if bridge:
        old_source = Path(registration["producer_source"])
        command += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=02:00:00",
            "--array=0-1%2",
        ]
        launcher = old_source / "mosaickv/slurm/evicastkv_review_closure_sequence.sbatch"
        args = [str(old_source), registration["old_root"], "bridge-worker"]
    else:
        command += ["--partition=bii", "--cpus-per-task=12", "--mem=16G", "--time=00:30:00"]
        launcher = SOURCE / (
            "mosaickv/slurm/evicastkv_review_closure_v2.sbatch"
            if tests
            else "mosaickv/slurm/evicastkv_review_source_recovery.sbatch"
        )
        args = (
            [str(SOURCE), str(root / "full-cpu"), "tests"]
            if tests
            else [str(SOURCE), str(root), mode]
        )
    if dependencies:
        command += ["--dependency=afterany:" + ":".join(identity(value) for value in dependencies)]
    command += [str(launcher), *args]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/{label}-request.json", dict(command=command, single_attempt=True))
    job = identity(subprocess.check_output(command, text=True))
    write_once(root / f"jobs/{label}-submission.json", dict(job=job, command=command))
    return job


def register(root, old=OLD, *, execute=False):
    root, old = safe_root(root), Path(old)
    if root.exists():
        raise FileExistsError("new immutable recovery root required")
    revision = clean_source(SOURCE)
    original = read(old / "registration.json")
    producer = Path(original["source_root"])
    clean_source(producer, OLD_SHA)
    if original["source_sha"] != OLD_SHA:
        raise ValueError("only the inspected V7 source is eligible")
    bindings = dict(original["bindings"])
    for relative in (
        "registration.json",
        "GATE_bridge_admission.json",
        "errors/analyze-source-prefix.json",
        "source-prefix/plan.json",
        "common-bridge/plan.json",
        "jobs/final-submission.json",
    ):
        path = old / relative
        bindings[str(path)] = sha(path)
    for index in range(2):
        path = old / f"source-prefix/workers/{index}/result.json"
        result = read(path)
        if result["status"] != "completed" or result["failures"] or result["source_sha"] != OLD_SHA:
            raise ValueError("two complete source workers required; no GPU retry authorized")
        if not {
            "raw.jsonl",
            "logits.pt",
            "manifest.json",
            "reference_prefix.json",
            "parity_controls.json",
        } <= set(result["outputs"]):
            raise ValueError("complete raw/logit/manifest/control output bindings required")
        bindings[str(path)] = sha(path)
        for name, expected in result["outputs"].items():
            bindings[str(path.parent / name)] = expected
    for name in ("evicastkv_review_source_recovery.py", "evicastkv_review_source_prefix.py"):
        path = SOURCE / "mosaickv/scripts" / name
        bindings[str(path)] = sha(path)
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable evidence changed: {path}")
    registration = dict(
        source_sha=revision,
        source_root=str(SOURCE),
        producer_source=str(producer),
        producer_sha=OLD_SHA,
        old_root=str(old),
        execute=execute,
        bindings=bindings,
        old_final_job=read(old / "jobs/final-submission.json")["job"],
        maximum_a100s=6,
        bridge_workers=2,
        source_gpu_retries=0,
        fresh_confirmation_authorized=False,
        new_cohort_authorized=False,
        original_negative_boundary_preserved=True,
    )
    write_once(root / "registration.json", registration)
    if not execute:
        return dict(status="dry_run", submitted_jobs=0)
    tests = submit(root, "cpu", "tests", tests=True)
    recovery = submit(root, "recovery", "recover", [tests])
    return dict(status="submitted", cpu=tests, recovery=recovery)


def load(root):
    record = read(root / "registration.json")
    clean_source(SOURCE, record["source_sha"])
    clean_source(Path(record["producer_source"]), record["producer_sha"])
    if record["execute"] is not True:
        raise ValueError("dry-run registration cannot execute")
    for path, expected in record["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable recovery binding changed: {path}")
    return record


def clone_controls(rows):
    selected = [row for row in rows if row["arm"] == "fullkv_forced_clone_control"]
    return len(selected) == 2 and all(
        row["status"] == "completed"
        and row["diagnostics"]["prefix_128"]["token_agreement"] == 1.0
        and row["diagnostics"]["prefix_128"]["maximum_logit_difference"] <= 0.001
        for row in selected
    )


def recovered_gate_allows_bridge(gate, rows):
    verifications = gate.get("recomputation_verifications", [])
    return (
        gate.get("complete") is True
        and gate.get("actual_rows") == 14
        and gate.get("completed_rows") == 14
        and not gate.get("failures")
        and gate.get("identical_reference_conditioning_verified") is True
        and clone_controls(rows)
        and len(verifications) == 14
        and all(v.get("exact_guard_booleans_checked", 0) >= 4 for v in verifications)
    )


def recover(root):
    from evicastkv_review_source_prefix import analyze

    record = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", cpu_gate(root, record["source_sha"]))
        old, target = Path(record["old_root"]), root / "source-prefix"
        target.mkdir(parents=True, exist_ok=False)
        # Preserve the original byte-for-byte plan hash used by every worker.
        with (
            (old / "source-prefix/plan.json").open("rb") as source,
            (target / "plan.json").open("xb") as destination,
        ):
            shutil.copyfileobj(source, destination)
        os.symlink(
            (old / "source-prefix/workers").resolve(), target / "workers", target_is_directory=True
        )
        analyze(target, producer_source=Path(record["producer_source"]))
        gate = read(target / "analysis/GATE_source_fixed_prefix.json")
        rows = [
            json.loads(line)
            for index in range(2)
            for line in (old / f"source-prefix/workers/{index}/raw.jsonl").read_text().splitlines()
        ]
        if not recovered_gate_allows_bridge(gate, rows):
            raise ValueError("recovered completeness or unchanged FullKV controls failed")
        snapshot = gpu_snapshot()
        dependencies = sorted(
            {item["job"].split("_")[0] for item in snapshot["active"] + snapshot["pending"]}
        )
        write_once(
            root / "GATE_bridge_admission.json",
            dict(
                status="passed",
                snapshot=snapshot,
                source_gate_sha256=sha(target / "analysis/GATE_source_fixed_prefix.json"),
                wait_for_existing_gpu_jobs=dependencies,
                maximum_a100s=6,
                bridge_workers=2,
                source_gpu_retries=0,
                numerical_guard_unchanged=True,
            ),
        )
        workers = submit(root, "bridge", "bridge-worker", dependencies, bridge=True)
        callback = submit(root, "bridge-analysis", "bridge-analyze", [workers])
        return dict(status="bridge_submitted", workers=workers, analysis=callback)
    except Exception as error:
        write_once(
            root / "recovery_error.json",
            dict(status="blocked", error=f"{type(error).__name__}: {error}"),
        )
        return dict(
            status="blocked", final=submit(root, "final", "finish", [record["old_final_job"]])
        )


def bridge_analyze(root):
    record = load(root)
    producer, old = Path(record["producer_source"]), Path(record["old_root"])
    environment = dict(
        os.environ, PYTHONPATH=f"{producer}/mosaickv/src:{producer}/mosaickv/scripts"
    )
    command = [
        PYTHON,
        str(producer / "mosaickv/scripts/evicastkv_review_common_bridge.py"),
        "analyze",
        "--root",
        str(old / "common-bridge"),
    ]
    with (root / "bridge-analysis.log").open("x") as log:
        result = subprocess.run(
            command,
            cwd=producer,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1500,
            check=False,
        )
    write_once(
        root / "bridge-analysis-result.json",
        dict(
            returncode=result.returncode,
            command=command,
            log_sha256=sha(root / "bridge-analysis.log"),
            producer_source=record["producer_sha"],
        ),
    )
    return dict(status="analyzed", final=submit(root, "final", "finish", [record["old_final_job"]]))


def finish(root):
    record = load(root)
    old = Path(record["old_root"])
    packet = dict(
        source_sha=record["source_sha"],
        producer_sha=record["producer_sha"],
        original_negative_boundary_preserved=True,
        source_GPU_retries=0,
        confirmation_authorized=False,
        superiority_established=False,
        artifacts={},
    )
    for path in (
        root / "source-prefix/analysis/GATE_source_fixed_prefix.json",
        root / "recovery_error.json",
        root / "bridge-analysis-result.json",
        root / "GATE_cpu_suite.json",
        root / "GATE_bridge_admission.json",
        old / "common-bridge/analysis/GATE_common_bridge.json",
        old / "FINAL_report.json",
    ):
        packet["artifacts"][str(path)] = (
            dict(path=str(path), sha256=sha(path), value=read(path))
            if path.is_file()
            else dict(status="missing_or_blocked")
        )
    write_once(root / "FINAL_report.json", packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "recover", "bridge-analyze", "finish"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=OLD)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, args.old_root, execute=args.execute)
        if args.mode == "register"
        else {"recover": recover, "bridge-analyze": bridge_analyze, "finish": finish}[args.mode](
            args.root
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
