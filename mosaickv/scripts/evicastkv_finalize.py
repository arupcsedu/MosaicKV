#!/usr/bin/env python3
"""Finite completion-triggered final audit. No GPU execution or label-based tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from evicastkv_campaign import SOURCE
from evicastkv_evidence import MATERIALIZATION
from evicastkv_packet import packet, read, text_once

from mosaickv.evaluation.fox_campaign import TASKS, clean_source, sha, write_once

ORDER = (
    "smoke",
    "representation_canary",
    "correctness",
    "scoring",
    "profiling",
    "measurement",
    "mechanisms",
    "full_graph_pilot",
)


def confirmation_feasibility(root, output):
    """Conservative identity-only upper bounds, not an assertion of fresh data."""
    path = root / "history-inventory.json"
    if not path.exists():
        write_once(
            output,
            dict(
                status="BLOCKED",
                reason="historical inventory missing",
                confirmation_submitted=False,
            ),
        )
        return
    inventory = read(path)
    meta = read(MATERIALIZATION)
    used = set(inventory["MileBench_consumed_ids"])
    counts = {}
    bindings = {str(path): sha(path), str(MATERIALIZATION): sha(MATERIALIZATION)}
    for task in TASKS:
        annotation = Path(meta["dependencies"][task]["annotation"])
        expected = meta["dependencies"][task]["annotation_sha256"]
        if sha(annotation) != expected:
            raise ValueError("pinned task identity source changed")
        bindings[str(annotation)] = expected
        identities = {f"{task}-{sample['sample_id']:04d}" for sample in read(annotation)["data"]}
        counts[task] = dict(
            official_total_ids=len(identities),
            known_used_ids=len(identities & used),
            maximum_potentially_unused_ids_before_media_checks=len(identities - used),
            required=64,
            id_count_feasible=len(identities - used) >= 64,
        )
    blockers = []
    if not inventory["complete_scan"]:
        blockers.append("historical ID/media inventory is not exhaustive")
    if not all(value["id_count_feasible"] for value in counts.values()):
        blockers.append("the four pinned MileBench subtasks cannot supply 64 unused IDs each")
    reviewed = root / "selected_config.json"
    if not reviewed.exists() or not read(reviewed).get("human_reviewed", False):
        blockers.append("no human-reviewed selected_config.json exists")
    write_once(
        output,
        dict(
            status="BLOCKED" if blockers else "REVIEW_REQUIRED_NOT_AUTOMATICALLY_AUTHORIZED",
            blockers=blockers,
            milebench_availability_upper_bounds=counts,
            reusable_media_disjointness_certified=False,
            requested_confirmation={"coco": 256, "milebench": 256},
            requested_primary_request_count=3584,
            actual_confirmation_requests=0,
            known_COCO_consumed_ids=len(inventory["COCO_consumed_ids"]),
            conservative_upper_bound_only=True,
            bound_files=bindings,
            confirmation_submitted=False,
            no_datasets_replaced=True,
            no_labels_used_for_selection=True,
        ),
    )


def finalize(primary):
    clean_source(SOURCE)
    output = primary / "writing/final-audit"
    output.mkdir(parents=True, exist_ok=False)
    confirmation_feasibility(primary, output / "GATE_fresh_confirmation_readiness.json")
    roots = [primary]
    child = primary.parent / (primary.name + "-mechanisms")
    if (child / "protocol_lock.json").exists():
        roots.append(child)
    results = []
    for index, root in enumerate(roots):
        destination = output / f"cohort-{index}"
        packet(root, destination)
        lock = read(root / "protocol_lock.json")
        results.append(
            dict(
                root=str(root),
                source_sha=lock["source_sha"],
                protocol_sha256=sha(root / "protocol_lock.json"),
                packet=str(destination / "final_writing_packet.md"),
                packet_manifest_sha256=sha(destination / "analysis_manifest.json"),
            )
        )
    write_once(
        output / "reproduction_index.json",
        dict(
            analysis_source_sha=clean_source(SOURCE),
            cohorts=results,
            complete_confirmation=False,
            headline_three_tier_claim_allowed=False,
            primary_and_mechanism_protocols_pooled=False,
        ),
    )
    lines = [
        "# Fox-review experimental campaign: final execution audit",
        "",
        "This is an execution ledger, not a promise that a method wins.",
        "",
        "The original dirty main checkout and historical raw results were preserved.",
        "No libraries were installed/upgraded. Model, data, output and byte controls were frozen.",
        "All caches and outputs use scratch. No user token was persisted.",
        "",
        "## Evidence packets",
        "",
        *[f"- `{item['packet']}` (runtime `{item['source_sha']}`)" for item in results],
        "",
        "## Confirmation boundary",
        "",
        "Read GATE_fresh_confirmation_readiness.json. The balanced 256+256 fresh confirmation "
        "must not be fabricated from consumed inputs or replacement datasets. Its 3,584 primary "
        "requests have not been launched.",
        "",
        "## Interpretation",
        "",
        "- Keep corpus CIDEr, Accuracy, and ROUGE-L in separate tables.",
        "- Equal initial capacity and matched append counts permit final/peak comparisons; "
        "it does not demonstrate a smaller-cache-at-equal-quality frontier.",
        "- Bonferroni intervals crossing zero are inconclusive, not equivalence.",
        "- Report actual negative controls, inactive graph sources, failed rows and denominators.",
        "- Reference-runtime profiling is not a verified optimization or a before/after speedup.",
        "- Exact-selection results cannot support prototype/residual/repair or serving claims.",
        "",
        "## Event execution",
        "",
        "Each finite callback reads one stage and schedules at most one bounded successor. "
        "No continuous polling or interactive-agent wake-up is claimed.",
        "Callbacks stop on failed gates and at the explicit configuration/freshness boundary.",
        "",
        "## Reproduction commands",
        "",
        "```bash",
        "source mosaickv/scripts/cache_env.sh",
        'export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""',
        'export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"',
        "# Use evicastkv_packet.py packet with a NEW output path to regenerate saved results.",
        "```",
        "",
        "SVG and TikZ/PGFPlots are code-native measured figures. No PDF compilation is claimed "
        "when no TeX engine is installed.",
    ]
    text_once(output / "README.md", "\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(status="finalized_with_explicit_gates", output=str(output), cohorts=results)
        )
    )


def follow(primary, root, stage):
    """Called only after the corresponding finite packet job has terminated."""
    clean_source(SOURCE)
    decision_path = root / "analysis" / stage / "completion_decision.json"
    if not decision_path.exists():
        finalize(primary)
        return
    decision = read(decision_path)
    successor = (
        decision.get("stage")
        if decision.get("action") == "submitted_registered_successor"
        else None
    )
    if stage == "measurement" and decision.get("action") == "stop_for_measured_hotspot_review":
        root = primary.parent / (primary.name + "-mechanisms")
        successor = (
            "mechanisms" if (root / "jobs/mechanisms-packet-submission.json").exists() else None
        )
    if successor is None:
        finalize(primary)
        return
    if successor not in ORDER or ORDER.index(successor) <= ORDER.index(stage):
        raise ValueError("completion callbacks may not loop or reorder stage gates")
    packet_job = root / "jobs" / f"{successor}-packet-submission.json"
    if not packet_job.exists():
        finalize(primary)
        return
    schedule(primary, root, successor, read(packet_job)["job"])


def schedule(primary, root, stage, dependency):
    source_sha = clean_source(SOURCE)
    registration = root / "jobs" / f"{stage}-final-audit-{source_sha[:12]}.json"
    if registration.exists():
        raise FileExistsError("this immutable completion callback is already registered")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:30:00",
        "--job-name=evicast-finalize",
        f"--dependency=afterany:{dependency}",
        f"--output={primary}/slurm/finalize-{stage}-%j.out",
        f"--error={primary}/slurm/finalize-{stage}-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_finalize.sbatch"),
        str(SOURCE),
        str(primary),
        str(root),
        stage,
        source_sha,
    ]
    jobid = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(registration, dict(job=jobid, command=command, source_sha=source_sha))
    print(json.dumps(dict(stage=stage, final_audit_job=jobid)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("schedule", "follow", "finalize"))
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage", choices=ORDER, default="correctness")
    parser.add_argument("--dependency")
    args = parser.parse_args()
    if args.mode == "finalize":
        finalize(args.primary)
    elif args.mode == "schedule":
        schedule(args.primary, args.root or args.primary, args.stage, args.dependency)
    else:
        follow(args.primary, args.root or args.primary, args.stage)


if __name__ == "__main__":
    main()
