#!/usr/bin/env python3
"""Reconcile a completed V36 audit without replaying scans or waiving errors.

This is a derived metadata analysis, not a fresh-cohort admission authority.
The model producer stays V36; this analyzer has a separate source identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import evicastkv_provenance_import as prior
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.history_occurrence_reconciliation import reconcile_occurrences

SOURCE = Path(__file__).resolve().parents[2]
BASE = "1e90ac0c1190809596a820c636b543473f55696f"
SCRATCH = Path("/scratch/djy8hg/workdir")
MAX_BYTES = 128 * 2**20


def metadata(path, bindings, expected=None):
    path = Path(path)
    if ".codex" in path.parts or path.name.lower() in {"auth.json", "credentials.json"}:
        raise ValueError("credential metadata bodies are outside reconciliation scope")
    path = path.resolve(strict=True)
    if ".codex" in path.parts or path.name.lower() in {"auth.json", "credentials.json"}:
        raise ValueError("credential aliases are outside reconciliation scope")
    if not path.is_relative_to(SCRATCH.resolve(strict=True)):
        raise ValueError("only project-owned review metadata may be read")
    raw, actual = read_bytes(path, bound=MAX_BYTES, expected=expected)
    bindings[str(path)] = actual
    return read_json(raw)


def source_bridge():
    changes = subprocess.check_output(
        ["git", "diff", "--name-status", f"{BASE}..HEAD"],
        cwd=SOURCE,
        text=True,
        timeout=15,
    )
    additions = []
    for line in changes.splitlines():
        action, name = line.split("\t", 1)
        if action != "A" or not name.startswith("mosaickv/"):
            raise ValueError("metadata reconciliation may not modify the frozen implementation")
        additions.append(name)
    return dict(model_source_sha=BASE, analyzer_additions=additions)


def checked_coverage(root, bindings, closeout_status):
    """Reuse successful bounded proofs, including when the wave was partial."""
    folder = root / "review/coverage"
    gate_path = folder / "GATE_coverage_resolution_v36.json"
    if not gate_path.is_file():
        return None, ["coverage_wave_missing_or_failed"]
    if "gate_sha256" not in closeout_status:
        return None, ["coverage_wave_not_qualified_by_closeout"]
    gate = metadata(gate_path, bindings, closeout_status["gate_sha256"])
    if gate != closeout_status["gate"]:
        raise ValueError("closeout and coverage wave decisions disagree")
    plan = metadata(folder / "plan.json", bindings)
    overlay = metadata(folder / "coverage_overlay.json", bindings, gate["coverage_overlay_sha256"])
    metadata(folder / "net_error_report.json", bindings, gate["net_error_report_sha256"])
    if gate["source_sha"] != BASE or plan["source_sha"] != BASE or overlay["source_sha"] != BASE:
        raise ValueError("coverage producer differs from the actual V36 source")
    for path, expected in plan["bindings"].items():
        canonical = Path(path).resolve(strict=True)
        if sha(canonical) != expected:
            raise ValueError("coverage parent/source binding changed")
        bindings[path] = expected
        bindings[str(canonical)] = expected
    worker_proofs = []
    for path, expected in gate["worker_bindings"].items():
        worker = metadata(path, bindings, expected)
        if worker["source_sha"] != BASE or worker["plan_sha256"] != sha(folder / "plan.json"):
            raise ValueError("coverage worker producer/plan binding differs")
        worker_proofs.extend(worker["committed"])
    if worker_proofs != overlay["new_proofs"]:
        raise ValueError("coverage overlay differs from actual committed worker proofs")
    blockers = []
    for name in ("missing_files", "pending_files", "blocked_files"):
        if gate[name]:
            blockers.append(f"coverage_{name}:{gate[name]}")
    return overlay, blockers


def imported_identity_proofs(verified, bindings):
    """Import already validated identity payloads; never replay YAML parsers."""
    old_plan = read_json(verified["raws"]["plan.json"])
    proofs, found = [], set()
    for name, expected in old_plan["bindings"].items():
        path = Path(name)
        # These are the exact hash-bound V12 proof workers, not source YAML.
        if path.parent.name != "workers" or path.parent.parent.name != "yaml":
            continue
        if path.name not in {f"shard-{index}.json" for index in range(8)}:
            continue
        record = metadata(path, bindings, expected)
        found.add(path.name)
        for proof in record["committed"]:
            if "identities" in proof:
                proofs.append(proof)
    if found != {f"shard-{index}.json" for index in range(8)}:
        raise ValueError("the eight preserved YAML identity-proof workers are required")
    return proofs


def imported_identity_inventories(verified, bindings):
    """Recover the identity payload linked by the reviewed oversize gate."""
    old_plan = read_json(verified["raws"]["plan.json"])
    candidates = [
        (Path(name), expected)
        for name, expected in old_plan["bindings"].items()
        if Path(name).name == "GATE_pending_oversize_recovery.json"
        and Path(name).parent.name == "oversize"
    ]
    if len(candidates) != 1:
        raise ValueError("one preserved oversize-identity gate is required")
    path, expected = candidates[0]
    gate = metadata(path, bindings, expected)
    inventory_path = path.parent / "combined_identity_inventory.json"
    inventory = metadata(inventory_path, bindings, gate["identity_inventory_sha256"])
    return [
        dict(
            path=str(inventory_path.resolve(strict=True)),
            sha256=gate["identity_inventory_sha256"],
            inventory=inventory,
        )
    ]


def analyze(history_root, output, *, cpu_gate_root):
    revision = clean_source(SOURCE)
    bridge = source_bridge()
    root, output = Path(history_root).resolve(strict=True), Path(output).resolve()
    if not root.is_relative_to(SCRATCH.resolve()) or not output.is_relative_to(SCRATCH.resolve()):
        raise ValueError("review inputs and derived outputs must remain in project scratch")
    if output.exists():
        raise FileExistsError("new immutable derived output directory required")
    bindings = {}
    cpu_root = Path(cpu_gate_root).resolve(strict=True)
    cpu = cpu_gate(cpu_root, revision)
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        path = cpu_root / "full-cpu" / name
        bindings[str(path)] = sha(path)
    registration = metadata(root / "registration.json", bindings)
    gate = metadata(root / "GATE_global_history_v36.json", bindings)
    inventory = metadata(
        root / "combined_identity_inventory.json", bindings, gate["inventory_sha256"]
    )
    closeout = metadata(root / "review/closeout.json", bindings)
    if any(record["source_sha"] != BASE for record in (registration, gate, inventory, closeout)):
        raise ValueError("actual V36 metadata producer identity must be retained")
    if closeout["global_history_gate"] != gate:
        raise ValueError("closeout and actual global audit decisions disagree")
    if gate["historical_coverage_errors"] != len(inventory["errors"]):
        raise ValueError("global raw-error denominator differs")
    verified = prior.validate_saved(prior.PARENT)
    old_plan = read_json(verified["raws"]["plan.json"])
    for path, expected in verified["bindings"].items():
        bindings[path] = expected
        bindings[str(Path(path).resolve(strict=True))] = expected
    original = metadata(
        old_plan["original_errors_path"],
        bindings,
        old_plan["bindings"][old_plan["original_errors_path"]],
    )
    identities = imported_identity_proofs(verified, bindings)
    prior_inventories = imported_identity_inventories(verified, bindings)
    overlay, blockers = checked_coverage(root, bindings, closeout["additive_coverage_recovery"])
    media = metadata(root / "review/media-gap.json", bindings, closeout["media_recovery_sha256"])
    report = reconcile_occurrences(
        inventory,
        seed_error_counts=registration["seed_error_counts"],
        preserved_resolved_error_sha256s=registration["preserved_resolved_error_sha256s"],
        coverage_overlay=overlay,
        original_errors=original,
        additional_identity_proofs=identities,
        additional_identity_inventories=prior_inventories,
        source_bindings=bindings,
    )
    if (
        report["counts"]["inherited_resolved_original_occurrences"]
        != gate["preserved_prior_resolutions"]
    ):
        raise ValueError("preserved prior-resolution denominator differs from global audit")
    if not gate["complete"] or gate["remaining_pending_items"]:
        blockers.append("global_traversal_or_current_time_verification_incomplete")
    if report["counts"]["net_unresolved_error_count"]:
        blockers.append("unresolved_historical_error_occurrences")
    # Neither a hash-bound inventory nor cleanup intent supplies missing media.
    if not media.get("historical_media_exposure_resolved", False):
        blockers.append("historical_media_source_group_exposure_not_certified")
    result = dict(
        analyzer_source_sha=revision,
        model_source_sha=BASE,
        source_bridge=bridge,
        cpu_gate=cpu,
        history_gate_complete=gate["complete"],
        source_bindings=bindings,
        reconciliation=report,
        blockers=blockers,
        status="derived_history_reconciliation_not_confirmation_admission",
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_claim_allowed=False,
        GPU_jobs_submitted=0,
        scientific_gates_unchanged=True,
    )
    write_once(output / "reconciled_history.json", result)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"reconciliation", "source_bindings"}
    }
    summary["reconciled_history_sha256"] = sha(output / "reconciled_history.json")
    summary["error_counts"] = report["counts"]
    summary["identity_counts"] = {
        name: len(values) for name, values in report["identities"].items()
    }
    summary["identity_addition_counts"] = report["identity_addition_counts"]
    write_once(output / "decision.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.history_root, args.output, cpu_gate_root=args.cpu_gate_root),
            sort_keys=True,
            allow_nan=False,
        )
    )
