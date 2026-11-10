#!/usr/bin/env python3
"""Read-only evidence inventory; execution completion never grants paper admission.

This exporter does not load predictions, answers, model weights, or a GPU. The
November 9 matrix remains an immutable historical assessment. Newly supplied
diagnostics extend the inventory, not its scientific authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = Path(__file__).resolve().parents[2]
HISTORICAL = SOURCE / "mosaickv/docs/fox_review_status_20261109.json"
RUNS = Path("/scratch/djy8hg/runs/evicastkv")
AUDITS = Path("/scratch/djy8hg/audits")
V4 = RUNS / "fox-review-v4-native-coco-20261109-mechanisms"
V5 = RUNS / "fox-review-v5-bounded-followups-20261109"
V6 = RUNS / "fox-review-v6-parallel-controls-20261110"
V7 = AUDITS / "fox-review-v7-closure-20261110"
RECOVERY = AUDITS / "fox-review-v7-source-analysis-recovery-20261110"
V8 = AUDITS / "fox-review-v8-consumed-diagnostics-20261110"

# Hashes pin completed parent packets; their nested references bind the gates.
SOURCES = {
    "mechanisms": V4 / "analysis/mechanisms/GATE_mechanisms.json",
    "full_routing": V4 / "analysis/full_graph_pilot/GATE_full_graph_pilot.json",
    "windows": V5 / "windows/analysis/window_sensitivity/GATE_window_sensitivity.json",
    "caption_audit": V5 / "captions/GATE_caption_audit.json",
    "v6_final": {
        "path": V6 / "FINAL_report.json",
        "sha256": "ef67e815d091c852f7c00fd32421ac7769b072b4080810ddb757fe1c594ad6e9",
    },
    "terms": V6 / "terms/analysis/mechanisms/GATE_independent_terms.json",
    "pooling": V6 / "pooling/analysis/GATE.json",
    "pooling_timing": V6 / "pooling/analysis/paired_timing.json",
    "replay": V6 / "source/analysis/GATE_source_controls.json",
    "v7_recovery": {
        "path": RECOVERY / "FINAL_report.json",
        "sha256": "f64b902277471b90a77e5705c833b3a7b673e3e0bf5d757f24195a3e0e02c5bb",
    },
    "source_prefix": RECOVERY / "source-prefix/analysis/GATE_source_fixed_prefix.json",
    "bridge": V7 / "common-bridge/analysis/GATE_common_bridge.json",
    "ownership": V7 / "ownership/analysis/GATE_ownership_closure.json",
    "v8_final": {
        "path": V8 / "FINAL_report.json",
        "sha256": "6b214b5035be34383c10e2fa611d52e9260ac5abf17fa69ff570b39869c276c5",
    },
    "native_norm": V8 / "rmsnorm/GATE_native_rmsnorm_replay.json",
    "protection": V8 / "protection/analysis/GATE_source_protection.json",
    "freshness": V8 / "freshness-continuation/aggregate/GATE_freshness_stream.json",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def referenced_hashes(value):
    """Only explicit artifact links, never interpret arbitrary strings as paths."""
    if isinstance(value, dict):
        path = value.get("path", value.get("gate_path"))
        expected = value.get("sha256", value.get("gate_sha256"))
        if path is not None and expected is not None:
            yield str(path), expected
        for key in ("bindings", "worker_bindings", "worker_summary_bindings"):
            links = value.get(key, {})
            if isinstance(links, dict):
                yield from links.items()
        # Some final packets use absolute artifact paths as mapping keys.
        for key, item in value.items():
            if isinstance(item, dict) and str(key).startswith("/") and "sha256" in item:
                yield str(key), item["sha256"]
            yield from referenced_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from referenced_hashes(item)


def inventory(sources):
    """Fail closed on missing/tampered evidence while retaining negative gates."""
    evidence, failures, seen = {}, [], {}

    def verify(path, expected=None):
        path = Path(path)
        canonical = str(path.resolve(strict=True))
        observed = sha(path)
        if canonical in seen and seen[canonical] != observed:
            raise ValueError(f"artifact changed during inventory: {path}")
        seen[canonical] = observed
        if expected is not None and observed != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
        return observed

    for name, spec in sources.items():
        spec = spec if isinstance(spec, dict) else {"path": spec}
        path = Path(spec["path"])
        try:
            observed = verify(path, spec.get("sha256"))
            value = (
                {"artifact_type": "hash_bound_document", "content_not_parsed": True}
                if spec.get("hash_only")
                else load(path)
            )
            if not isinstance(value, (dict, list)):
                raise ValueError("artifact must be a JSON object or table")
            for linked, expected in referenced_hashes(value):
                verify(linked, expected)
            # Detect a producer modifying a file during this read.
            if sha(path) != observed:
                raise ValueError(f"artifact changed while reading: {path}")
            evidence[name] = {
                "path": str(path),
                "sha256": observed,
                "verified": True,
                "value": value,
            }
        except (OSError, ValueError, TypeError) as error:
            failures.append({"evidence_id": name, "path": str(path), "error": str(error)})
            evidence[name] = {"path": str(path), "verified": False, "error": str(error)}
    return evidence, failures


def observation(evidence, name, *keys):
    item = evidence.get(name, {})
    if not item.get("verified"):
        return None
    value = item["value"]
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def build_matrix(historical, evidence, failures, *, diagnostics_present=False):
    """Export separate engineering, measurement, and inference states for R01-10."""
    summary = {
        "mechanism_rows": observation(
            evidence, "mechanisms", "tests_actually_executed", "completed"
        ),
        "full_routing_rows": observation(
            evidence, "full_routing", "tests_actually_executed", "completed"
        ),
        "window_rows": observation(evidence, "windows", "actual_rows"),
        "independent_term_rows": observation(evidence, "terms", "actual_rows"),
        "pooling_rows": observation(evidence, "pooling", "rows"),
        "pooling_equivalence_checks": observation(evidence, "pooling", "checks"),
        "source_fixed_prefix_rows": observation(evidence, "source_prefix", "actual_rows"),
        "common_model_bridge_rows": observation(evidence, "bridge", "rows"),
        "ownership_certificates": observation(evidence, "ownership", "reports"),
        "protection_capture_rows": observation(evidence, "protection", "rows"),
        "all_base_source_hashes_equal": observation(
            evidence, "protection", "all_base_source_hashes_equal"
        ),
        "all_protected_rows_bitwise_original": observation(
            evidence, "protection", "all_protected_rows_bitwise_original"
        ),
        "source_all_registered_guards_pass": observation(
            evidence, "source_prefix", "all_source_variants_pass_registered_guard"
        ),
        "bridge_realized_bytes_matched": observation(
            evidence, "bridge", "shared_realized_byte_comparison"
        ),
        "exact_owner_peaks_certified": observation(
            evidence, "ownership", "all_six_exact_KV_peaks_certified"
        ),
        "whole_device_peak_certified": observation(
            evidence, "ownership", "whole_device_memory_certified"
        ),
        "historical_full_model_norm_parity": observation(
            evidence, "native_norm", "full_model_parity_established"
        ),
        "historical_media_inventory_complete": observation(
            evidence, "freshness", "complete_historical_id_inventory"
        ),
        "cross_dataset_media_certified": observation(
            evidence, "freshness", "cross_dataset_media_certified"
        ),
    }
    timing = observation(evidence, "pooling_timing")
    summary["pooling_timing"] = [
        {
            "method": row.get("method"),
            "statistics": row.get("paired_statistics"),
            "scope": "four-consumed-input historical paired diagnostic, not confirmation",
        }
        for row in timing or []
    ]
    specs = {
        "R01": (
            "Runtime activity instrumentation implemented for the exact-selection variant.",
            ["mechanisms", "full_routing", "terms"],
            "Completed runtime activity is measured; it does not activate the "
            "source variant's six mechanisms.",
            ["Do not pool exact selection, shared-source storage, and the six-component hybrid."],
        ),
        "R02": (
            "Original-position, packed-subset, native/clone and operator-level "
            "diagnostics implemented.",
            ["native_norm", "protection", "source_prefix"],
            "Native RMSNorm shape dependence is reproduced at operator level; historical "
            "full-model controls and lossy-source fidelity remain distinct negative findings.",
            [
                "A successful operator replay is not a full-model arithmetic correction.",
                "Optional V9 numerical diagnostics must be assessed separately without "
                "rewriting the historical failures.",
            ],
        ),
        "R03": (
            "Native scoring and saved-caption audits implemented; local baseline labels retained.",
            ["caption_audit", "v6_final"],
            "Saved scoring/audit evidence exists; no unqualified official LOOK-M/VL-Cache "
            "or general baseline-superiority conclusion follows.",
            [
                "Qualify pinned baseline fidelity and retain Accuracy, ROUGE-L, PPL, "
                "CIDEr and ANLS as separate metrics."
            ],
        ),
        "R04": (
            "Complete-path profiling and equivalent pooled-array optimization implemented.",
            ["pooling", "pooling_timing"],
            "Historical pooling timing is a consumed-input microstudy, not a current "
            "full-model norm control or independent latency claim.",
            [
                "Freeze any promoted execution policy before independent quality "
                "and paired latency validation."
            ],
        ),
        "R05": (
            "Mechanism, observation-window, full-routing and independent-term "
            "ablations implemented.",
            ["mechanisms", "full_routing", "windows", "terms"],
            "Completed exploratory ablations include null/negative effects; completed "
            "rows do not establish useful effects of every mechanism.",
            [
                "Do not infer six-component performance from exact-only ablations or "
                "choose favorable exploratory points as confirmation."
            ],
        ),
        "R06": (
            "Same-model source replay OFF/ON and native/forced-prefix controls implemented.",
            ["replay", "source_prefix"],
            "Replay execution and numerical controls are measured on consumed inputs; "
            "shared-prefix conditioning is not independent free-generation quality.",
            ["Keep replay/storage timing and exact-selection timing scopes distinct."],
        ),
        "R07": (
            "Direct projected-K/V control and common-BF16-model bridge implemented.",
            ["source_prefix", "bridge", "protection"],
            "The common-model bridge executes but common ceilings are not equal "
            "realized storage; source guard failures remain negative.",
            [
                "Run genuinely realized-byte-matched curves only after source fidelity "
                "and feasible geometry are established.",
                "KIVI/KVQuant/ZipCache and official-baseline claims require audited "
                "pinned implementations.",
            ],
        ),
        "R08": (
            "Bounded historical identity/media inventory and continuation tooling implemented.",
            ["freshness", "v8_final"],
            "Current cohorts are consumed development data; incomplete identity/media "
            "coverage blocks freshness and independent confirmation.",
            [
                "Finish historical identity and cross-dataset underlying-media/"
                "source-group coverage.",
                "Freeze a candidate and contrast family, then run powered independent validation.",
            ],
        ),
        "R09": (
            "Pointer-generation-aware ownership closure and scoped exact-KV "
            "certification implemented.",
            ["ownership", "bridge"],
            "Saved exact-selection owner peaks are certified only within their explicit "
            "boundary, not whole-device HBM or source-variant peaks.",
            [
                "Do not transfer exact-selection ownership certificates to the "
                "BF16 source variant.",
                "Source GQA geometry and context-length scaling remain unvalidated.",
            ],
        ),
        "R10": (
            "Hash-linked derived exports implemented; manuscript review remains required.",
            ["v6_final", "v7_recovery", "v8_final"],
            "Engineering closure does not establish superiority or author approval. "
            "Manuscript v1.8's LLM-use denial conflicts with documented assistance.",
            [
                "Correct the LLM Usage Statement after all-author review; do not "
                "certify approval automatically.",
                "Preserve negative results, historical scopes, complete denominators "
                "and accurate variant labels in every table.",
            ],
        ),
    }
    old = {row["id"]: row for row in historical["requirements"]}
    rows = []
    for identity, (implemented, names, conclusion, remaining) in specs.items():
        verified = [name for name in names if evidence.get(name, {}).get("verified")]
        rows.append(
            {
                "id": identity,
                "requirement": old[identity]["requirement"],
                "implementation_status": implemented,
                "measurement_status": "evidence_verified_scoped_observations"
                if len(verified) == len(names)
                else "blocked_missing_or_unverified_evidence",
                "evidence_ids": names,
                "verified_evidence_ids": verified,
                "scientific_conclusion": conclusion,
                "remaining": remaining,
                "paper_admissibility": "not_established_by_this_inventory",
            }
        )
    current = {
        name: item["value"]
        for name, item in evidence.items()
        if name.startswith("current_") and item.get("verified")
    }
    rows[1]["evidence_ids"].extend(current)
    rows[1]["verified_evidence_ids"].extend(current)
    return {
        "schema_version": 2,
        "assessment_date": "2026-11-10",
        "assessment_kind": "derived_evidence_inventory_not_new_experiments",
        "system_name": "EviCastKV",
        "historical_system_name": "MosaicKV",
        "superiority_established": False,
        "fresh_confirmation_authorized": False,
        "paper_campaign_authorized": False,
        "candidate_freeze_authorized": False,
        "new_GPU_inference": 0,
        "benchmark_answers_read": False,
        "historical_negative_controls_preserved": True,
        "current_diagnostics_status": "provided_separately_not_automatic_admission"
        if diagnostics_present
        else "pending_not_supplied",
        "current_diagnostic_observations": current,
        "summary": summary,
        "requirements": rows,
        "evidence_failures": failures,
        "scope": historical.get("scope", {}),
        "proposed_LLM_disclosure_for_author_review": (
            "LLM-based coding assistants contributed to research planning, repository inspection, "
            "implementation and testing, job orchestration, result analysis, "
            "and manuscript drafting. "
            "The authors are responsible for verifying the code, experimental evidence, scientific "
            "interpretation, citations, and final text. This proposed statement "
            "requires all-author review."
        ),
        "author_review_or_approval_certified": False,
    }


def markdown(matrix):
    lines = [
        "# Prof. Fox review evidence status — November 10, 2026",
        "",
        "Derived inventory only; no new experiment, superiority, freshness, freeze, "
        "or paper-admission authorization.",
        "",
        "Historical November 9 matrix and all negative controls are preserved.",
        "",
        "| Review | Implemented | Measured | Scientific conclusion | Evidence |",
        "|---|---|---|---|---|",
    ]
    for row in matrix["requirements"]:

        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")

        refs = ", ".join(f"`{name}`" for name in row["evidence_ids"])
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        row["id"],
                        row["implementation_status"],
                        row["measurement_status"],
                        row["scientific_conclusion"],
                        refs,
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Measurements derived from linked gates",
        "",
        "```json",
        json.dumps(matrix["summary"], indent=2, allow_nan=False),
        "```",
        "",
        "## Current diagnostics — distinct from historical controls",
        "",
        "```json",
        json.dumps(matrix["current_diagnostic_observations"], indent=2, allow_nan=False),
        "```",
        "",
        "## Evidence links",
        "",
    ]
    for name, item in matrix["evidence"].items():
        lines.append(
            f"- `{name}`: `{item['path']}`; SHA-256 "
            f"`{item.get('sha256', 'unverified')}`; verified={item['verified']}."
        )
    lines += ["", "## Remaining review work", ""]
    for row in matrix["requirements"]:
        lines += [f"- {row['id']}: {text}" for text in row["remaining"]]
    lines += [
        "",
        "## Proposed disclosure — requires all-author review",
        "",
        matrix["proposed_LLM_disclosure_for_author_review"],
        "",
        "No author review or approval is certified by this exporter.",
        "",
    ]
    return "\n".join(lines)


def run(root, *, current_diagnostics=None):
    root = Path(root)
    if root.exists():
        raise FileExistsError("a new write-once derived root is required")
    historical_hash = sha(HISTORICAL)
    historical = load(HISTORICAL)
    sources = dict(SOURCES)
    if current_diagnostics is not None:
        if not isinstance(current_diagnostics, dict) or not current_diagnostics:
            raise ValueError(
                "current_diagnostics must map diagnostic names to explicit artifact paths"
            )
        for name, spec in current_diagnostics.items():
            if not isinstance(name, str) or not name or name in sources:
                raise ValueError("current diagnostic names must be unique")
            sources[f"current_{name}"] = spec
    evidence, failures = inventory(sources)
    # Verify the original immutable evidence inventory without copying raw values.
    old_sources = {
        f"historical_{name}": dict(item, hash_only=Path(item["path"]).suffix != ".json")
        for name, item in historical["evidence"].items()
    }
    old_evidence, old_failures = inventory(old_sources)
    evidence.update(old_evidence)
    failures += old_failures
    matrix = build_matrix(
        historical, evidence, failures, diagnostics_present=current_diagnostics is not None
    )
    if sha(HISTORICAL) != historical_hash:
        raise ValueError("historical matrix changed while reading")
    matrix["historical_matrix"] = {
        "path": str(HISTORICAL),
        "sha256": historical_hash,
        "modified": False,
    }
    matrix["exporter"] = {"path": str(Path(__file__)), "sha256": sha(__file__)}
    matrix["evidence"] = {
        name: {k: v for k, v in item.items() if k != "value"} for name, item in evidence.items()
    }
    root.mkdir(parents=True)
    write_once(root / "review_matrix.json", matrix)
    with (root / "review_matrix.md").open("x") as stream:
        stream.write(markdown(matrix))
    gate = {
        "status": "passed_inventory_export_not_paper_admission"
        if not failures
        else "blocked_unverified_evidence",
        "requirement_rows": len(matrix["requirements"]),
        "evidence_artifacts": len(evidence),
        "evidence_failures": failures,
        "review_matrix_sha256": sha(root / "review_matrix.json"),
        "markdown_sha256": sha(root / "review_matrix.md"),
        "historical_matrix_sha256": historical_hash,
        "superiority_established": False,
        "fresh_confirmation_authorized": False,
        "candidate_freeze_authorized": False,
        "paper_campaign_authorized": False,
        "new_GPU_inference": 0,
        "author_review_or_approval_certified": False,
    }
    write_once(root / "GATE_review_status.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--current-diagnostics",
        type=Path,
        help="JSON mapping of diagnostic names to artifact paths/hash records",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.root,
                current_diagnostics=load(args.current_diagnostics)
                if args.current_diagnostics
                else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
