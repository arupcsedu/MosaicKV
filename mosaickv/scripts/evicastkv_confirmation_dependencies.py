#!/usr/bin/env python3
"""Bind confirmation dependencies without loading inputs or submitting jobs.

This is the CPU preparation entrypoint, not an experiment authorization. Only
explicit, hash-bound, bounded metadata capsules are read; their raw-data links
are never followed. An eligible plan still requires a separate execution scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, write_once
from mosaickv.evaluation.fox_confirmation_candidate import resolve_candidate_dependency
from mosaickv.evaluation.fox_confirmation_inputs import resolve_input_dependencies
from mosaickv.evaluation.fox_confirmation_plan import build_confirmation_plan
from mosaickv.evaluation.paper_readiness_v63 import ANCHOR, CapsuleReader

SOURCE = Path(__file__).resolve().parents[2]
REQUEST_SCHEMA = "evicastkv_confirmation_dependency_request_v1"
OPTIONAL_CAPSULES = (
    "frozen",
    "review",
    "processor",
    "confirmation_evidence",
    "readiness_capsule",
)
PREPARATION_SOURCE_FILES = (
    "mosaickv/scripts/evicastkv_confirmation_dependencies.py",
    "mosaickv/scripts/evicastkv_campaign.py",
    "mosaickv/scripts/evicastkv_worker.py",
    "mosaickv/slurm/evicastkv_worker.sbatch",
    "mosaickv/src/mosaickv/evaluation/fox_confirmation_candidate.py",
    "mosaickv/src/mosaickv/evaluation/fox_confirmation_inputs.py",
    "mosaickv/src/mosaickv/evaluation/fox_confirmation_plan.py",
)


def preparation_source_binding(source):
    """Identify scoped dirty preparation code, not a canonical experiment SHA."""
    rows, total = [], 0
    for relative in PREPARATION_SOURCE_FILES:
        path = Path(source) / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular scoped preparation source required")
        before = path.stat()
        total += before.st_size
        if before.st_size > 512 * 1024**2 or total > 8 * 1024**2:
            raise ValueError("bounded scoped source snapshot required")
        data = path.read_bytes()
        after = path.stat()
        if len(data) != before.st_size or any(
            getattr(before, k) != getattr(after, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError("preparation source changed during binding")
        rows.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return dict(
        scope="seven_preparation_and_guard_paths_not_full_repository",
        content_sha256=hashlib.sha256(content).hexdigest(),
        files=rows,
    )


def relative_path(path: str, anchor: Path) -> str:
    """Normalize only the known scratch alias; never resolve an arbitrary link."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("explicit capsule path required")
    value = Path(path)
    if ".." in value.parts:
        raise ValueError("capsule traversal forbidden")
    if value.is_absolute():
        alias = Path("/scratch/djy8hg/workdir")
        if anchor == ANCHOR and value.is_relative_to(alias):
            value = anchor / value.relative_to(alias)
        if not value.is_relative_to(anchor):
            raise ValueError("capsule outside registered metadata anchor")
        value = value.relative_to(anchor)
    if value.suffix != ".json" or value.name in {"raw.json", "auth.json"}:
        raise ValueError("named metadata JSON capsule required, not raw input")
    return str(value)


def read_capsule(reader, binding):
    if (
        not isinstance(binding, dict)
        or set(binding) != {"path", "sha256"}
        or not isinstance(binding.get("sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", binding["sha256"]) is None
    ):
        raise ValueError("exact metadata path/SHA256 binding required")
    record = reader.read(relative_path(binding["path"], reader.anchor), binding["sha256"])
    return dict(binding, record=record)


def resolve_request(reader, request, *, source_sha, source_clean):
    allowed = {
        "schema",
        "protocol",
        "config",
        "route_sources",
        "proof_capsules",
        *OPTIONAL_CAPSULES,
    }
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("registered confirmation dependency request required")
    if set(request) - allowed:
        raise ValueError("unregistered request fields cannot grant execution scope")
    protocol = read_capsule(reader, request["protocol"])
    config = read_capsule(reader, request["config"])
    optional = {
        name: read_capsule(reader, request[name]) if request.get(name) is not None else None
        for name in OPTIONAL_CAPSULES
    }
    raw_proofs = request.get("proof_capsules", {})
    if not isinstance(raw_proofs, dict) or set(raw_proofs) - {
        "correctness",
        "metric",
        "byte_matching",
        "baseline_fidelity",
        "history",
        "media",
    }:
        raise ValueError("only the six registered proof kinds are accepted")
    proofs = {name: read_capsule(reader, value) for name, value in raw_proofs.items()}
    candidate = resolve_candidate_dependency(
        protocol["record"],
        config["record"],
        config_file_sha256=config["sha256"],
        frozen=optional["frozen"]["record"] if optional["frozen"] else None,
        review=optional["review"],
        proof_capsules={
            k: v
            for k, v in proofs.items()
            if k in {"correctness", "metric", "byte_matching", "baseline_fidelity"}
        },
        expected_source_sha=source_sha,
    )
    inputs = resolve_input_dependencies(
        protocol["record"],
        route_sources=request.get("route_sources", {}),
        processor_descriptor=optional["processor"]["record"] if optional["processor"] else None,
    )
    evidence = optional["confirmation_evidence"]
    if evidence:
        evidence = dict(
            evidence["record"], proof_capsules=proofs, review_capsule=optional["review"]
        )
    plan = build_confirmation_plan(
        protocol["record"],
        candidate,
        inputs,
        confirmation_evidence=evidence,
        source_sha=source_sha,
        source_clean=source_clean,
        readiness_capsule=(
            optional["readiness_capsule"]["record"] if optional["readiness_capsule"] else None
        ),
    )
    return dict(
        plan=plan,
        candidate=candidate,
        inputs=inputs,
        confirmation_cohort=(evidence.get("cohort") if evidence else None),
        source_method_bindings=protocol["record"]["source_references"]["configs"],
        protocol_binding={k: protocol[k] for k in ("path", "sha256")},
        evidence_bindings=list(reader.bindings),
    )


def prepare(root: Path, request_path: str, request_sha256: str, *, anchor=ANCHOR, source=SOURCE):
    """Publish a new derived metadata packet, never a selected_config or job."""
    if Path(root).exists():
        raise FileExistsError("dependency packets are write-once; use a new output directory")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=source, text=True
        )
    )
    reader = CapsuleReader(Path(anchor))
    request = read_capsule(reader, {"path": request_path, "sha256": request_sha256})
    result = resolve_request(reader, request["record"], source_sha=revision, source_clean=not dirty)
    result.update(
        schema="evicastkv_confirmation_dependency_packet_v1",
        source_sha=revision,
        source_dirty=dirty,
        source_content_binding=preparation_source_binding(source),
        classification="CPU_dependency_preparation_not_experiment_results",
        request_binding={"path": request_path, "sha256": request_sha256},
        resource_usage=dict(
            metadata_input_bytes=reader.input_bytes,
            projected_metadata_bytes=reader.projected_bytes,
            metadata_nodes=reader.nodes,
        ),
        completed_proofs_replayed=0,
        original_raw_inputs_opened=0,
        model_or_dataset_accessed=False,
        jobs_submitted=0,
        GPU_jobs=0,
        implemented_entrypoints={
            "CPU_preparation": "mosaickv/scripts/evicastkv_campaign.py confirmation",
            "allocated_worker": (
                "mosaickv/scripts/evicastkv_campaign.py worker --stage confirmation"
            ),
            "admission_function": "evicastkv_confirmation_dependencies.admit_confirmation_worker",
            "submission": "registered_shared_scheduler_controller_and_separate_scope_required",
        },
    )
    write_once(Path(root) / "confirmation_dependencies.json", result)
    return result


def admit_confirmation_worker(root, spec_sha256, index, *, anchor=ANCHOR, source=SOURCE):
    """Check every dependency and separate scope BEFORE any model/data import.

    Nothing in preparation creates the execution-scope record this requires.
    Completed certificates are read as bounded metadata, never recomputed.
    """
    if type(index) is not int or index < 0:
        raise ValueError("registered nonnegative worker index required")
    revision = clean_source(source)
    reader = CapsuleReader(Path(anchor))
    spec = read_capsule(
        reader, {"path": str(Path(root) / "jobs/confirmation.json"), "sha256": spec_sha256}
    )["record"]
    packet = read_capsule(reader, spec.get("confirmation_dependency_binding"))
    request = read_capsule(reader, packet["record"].get("request_binding"))
    current = resolve_request(reader, request["record"], source_sha=revision, source_clean=True)
    if current["plan"]["prerequisite_metadata_validated"] is not True:
        raise ValueError("confirmation dependencies remain blocked; no model/data import allowed")
    for key in (
        "plan",
        "candidate",
        "inputs",
        "confirmation_cohort",
        "protocol_binding",
        "source_method_bindings",
    ):
        if current[key] != packet["record"].get(key):
            raise ValueError("prepared confirmation dependencies changed")
    scope = read_capsule(reader, spec.get("execution_scope_binding"))["record"]
    required = {
        "schema": "evicastkv_confirmation_execution_scope_v1",
        "source_sha": revision,
        "dependency_packet_sha256": packet["sha256"],
        "dependency_request_sha256": request["sha256"],
        "protocol_lock_sha256": spec.get("protocol_sha256"),
        "scope": "COCO_confirmation_only_no_followup",
        "explicit_execution_reviewed": True,
        "fresh_confirmation_authorized": True,
        "GPU_execution_authorized": True,
        "experiment_submission_authorized": True,
        "exclusion_scan_authorized": False,
        "materialization_authorized": False,
        "paper_admission_authorized": False,
        "automatic_stage_progression": False,
    }
    if any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in required.items()):
        raise ValueError("separate source/packet-bound confirmation-only scope required")
    jobs = confirmation_jobs()
    if spec.get("stage") != "confirmation" or spec.get("jobs") != jobs or index >= len(jobs):
        raise ValueError("confirmation job matrix changed")
    lock = read_capsule(
        reader,
        {"path": str(Path(root) / "protocol_lock.json"), "sha256": spec.get("protocol_sha256")},
    )["record"]
    if spec["protocol_sha256"] != current["plan"]["reviewed_runtime_lock_sha256"]:
        raise ValueError("runtime lock differs from the independently reviewed exact bytes")
    cohort = current["confirmation_cohort"]
    registered = lock.get("cohorts", {}).get("coco", {}).get("confirmation", {})
    if (
        lock.get("source_sha") != revision
        or lock.get("source_root") != str(source)
        or lock.get("source_patch_sha256") != hashlib.sha256(b"").hexdigest()
        or lock.get("enabled_workloads") != ["coco"]
        or lock.get("budgets_mib") != {"coco": [20, 40]}
        or type(lock.get("seed")) is not int
        or lock["seed"] != 20271014
        or type(lock.get("warmups")) is not int
        or lock["warmups"] != 5
        or type(lock.get("repeats")) is not int
        or lock["repeats"] != 3
        or set(lock.get("cohorts", {})) != {"coco"}
        or set(lock["cohorts"]["coco"]) != {"confirmation"}
        or set(lock.get("templates", {})) != {"coco"}
        or lock.get("authorization", {}).get("automatic_stage_progression") is not False
        or lock.get("optimization_protocol")
        or lock.get("inherited_gates")
        or registered.get("sha256") != cohort["ledger_sha256"]
        or registered.get("sample_ids") != cohort["sample_ids"]
        or relative_path(registered.get("path"), reader.anchor) != cohort["ledger_path"]
        or lock.get("confirmation_timing_sample_ids") != cohort["timing_sample_ids"]
        or lock.get("templates", {}).get("coco", {}).get("sha256")
        != current["candidate"]["candidate_config_file_sha256"]
    ):
        raise ValueError("runtime lock differs from qualified confirmation bindings")
    source_configs = {
        str(source / value["path"]): value["file_sha256"]
        for value in current["source_method_bindings"].values()
    }
    if lock.get("historical_bindings") != source_configs:
        raise ValueError("confirmation may bind only the four registered static source configs")
    resources = lock.get("resources", {})
    expected = {"cpus_per_worker": 12, "memory_gib_per_worker": 32, "wall_minutes": 120}
    if any(type(resources.get(k)) is not int or resources[k] != v for k, v in expected.items()):
        raise ValueError("confirmation resource limits changed")
    maximum = resources.get("maximum_a100s")
    if type(maximum) is not int or not 1 <= maximum <= 6:
        raise ValueError("confirmation aggregate A100 ceiling changed")
    return current


def reject_confirmation_relabel(root, stage, spec_sha256, index):
    """A development stage cannot sidestep admission by renaming a fresh job."""
    if not isinstance(stage, str) or not re.fullmatch("[a-z_]+", stage):
        raise ValueError("invalid stage name")
    if type(index) is not int or index < 0:
        raise ValueError("registered nonnegative worker index required")
    reader = CapsuleReader(Path(root))
    spec = read_capsule(reader, {"path": f"jobs/{stage}.json", "sha256": spec_sha256})["record"]
    lock = read_capsule(
        reader, {"path": "protocol_lock.json", "sha256": spec.get("protocol_sha256")}
    )["record"]
    jobs = spec.get("jobs")
    if not isinstance(jobs, list) or index >= len(jobs):
        raise ValueError("registered worker index outside job specification")
    if (
        spec.get("stage") == "confirmation"
        or any(job.get("subset") == "confirmation" for job in jobs)
        or any("confirmation" in subsets for subsets in lock.get("cohorts", {}).values())
    ):
        raise ValueError("confirmation cohorts require their separately admitted stage")


def confirmation_jobs():
    """Prospective immutable shards, not a record of submitted/completed jobs."""
    arms = [("full_kv", None, "complete")] + [
        (method, cap, "complete")
        for cap in (20, 40)
        for method in ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
    ]
    return [
        dict(
            workload="coco",
            subset="confirmation",
            stage="confirmation",
            indices=list(range(start, start + 8)),
            arms=[list(a) for a in arms],
        )
        for start in range(0, 256, 8)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    result = prepare(args.root, args.request, args.request_sha256)
    print(
        json.dumps(
            dict(
                status=result["plan"]["status"],
                output=str(args.root / "confirmation_dependencies.json"),
                blockers=result["plan"].get("blockers", []),
                jobs_submitted=0,
                GPU_jobs=0,
            ),
            sort_keys=True,
        )
    )
    return 0  # Successful dependency reporting is not an empirical gate pass.


if __name__ == "__main__":
    raise SystemExit(main())
